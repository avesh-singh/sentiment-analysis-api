from model import *
from data import prepare_data_keras
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from constants import *


train_buck, valid_buck, test_buck, valid_weights = prepare_data_keras()
model = Model(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, 1, LIN_DROPOUT).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())


def train(batch):
    model.zero_grad()
    label = batch[1].view(-1, 1).to(dtype=torch.float)
    text = batch[0]
    output = model(text)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    return loss.item() / len(batch), list(label.squeeze().cpu().numpy()), list(map(int, output > 0))


def train_iters():
    best_valid_score = 0
    for e in range(EPOCHS):
        train_loss = 0
        train_acc = 0
        train_tgt = []
        train_pred = []
        for i, batch in enumerate(train_buck):
            loss, train_target, train_prediction = train(batch)
            train_loss += loss
            train_tgt.extend(train_target)
            train_pred.extend(train_prediction)
        valid_loss, expected, prediction = evaluate(valid_buck)
        valid_score = accuracy_score(expected, prediction)
        if best_valid_score < valid_score:
            print("new best model! improvement: %f" % (valid_score - best_valid_score))
            best_valid_score = valid_score
            torch.save(model.state_dict(), 'model.pt')
        print(f"epochs: {e} | train acc: {accuracy_score(train_tgt, train_pred):.4f} | training precision:"
              f" {precision_score(train_tgt, train_pred):.4f} | "
              f"training "
              f"recall:"
              f" {recall_score(train_tgt, train_pred):.4f} | training loss: {train_loss / len(train_buck):.4f}")
        print(f"validation acc: {accuracy_score(expected, prediction):.4f} | validation precision:"
              f" {precision_score(expected, prediction):.4f} | validation "
              f"recall: "
              f"{recall_score(expected, prediction):.4f} | "
              f"validation score: {f1_score(expected, prediction):.4f} | "
              f"validation loss: {valid_loss:.4f}\n")
        print()


def evaluate(iterator):
    with torch.no_grad():
        all_targets = []
        all_preds = []
        loss = 0
        for b in iterator:
            label = b[1].view(-1, 1).to(dtype=torch.float)
            all_targets.extend(list(label.squeeze().cpu().numpy()))
            text = b[0]
            output = model(text)
            loss += criterion(output, label)
            pred = list(map(int, output > 0))
            all_preds.extend(pred)
    return loss / len(valid_buck), all_targets, all_preds


if __name__ == '__main__':
    train_iters()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    loss, tgts, preds = evaluate(test_buck)
    print(confusion_matrix(tgts, preds))
    print(f"test acc: {accuracy_score(tgts, preds):.4f} | test precision:"
          f" {precision_score(tgts, preds):.4f} | test "
          f"recall: "
          f"{recall_score(tgts, preds):.4f} | "
          f"test score: {f1_score(tgts, preds):.4f} | "
          f"test loss: {loss:.4f}\n")
