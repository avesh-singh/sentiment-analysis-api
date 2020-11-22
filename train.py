from model import *
from data import prepare_data_keras
from sklearn.metrics import precision_score, recall_score, f1_score
from constants import *


train_buck, valid_buck, test_buck, valid_weights = prepare_data_keras()
model = Model(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, 1, DROPOUT, N_LAYERS).to(device)
criterion = nn.BCEWithLogitsLoss()
# validation_criterion = nn.BCEWithLogitsLoss(pos_weight=valid_weights[1])
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
    best_valid_loss = float('inf')
    for e in range(EPOCHS):
        train_loss = 0
        train_tgt = []
        train_pred = []
        for i, batch in enumerate(train_buck):
            loss, train_target, train_prediction = train(batch)
            train_loss += loss
            train_tgt.extend(train_target)
            train_pred.extend(train_prediction)
        valid_loss, expected, prediction = evaluate()
        if best_valid_loss >= valid_loss:
            print("new best model! improvement: %f" % (best_valid_loss - valid_loss))
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')
        print(f"epochs: {e} | training precision: {precision_score(train_tgt, train_pred):.4f} | training recall:"
              f" {recall_score(train_tgt, train_pred):.4f} | training loss: {train_loss / len(train_buck):.4f} |"
              f" validation precision: {precision_score(expected, prediction):.4f} | validation recall: "
              f"{recall_score(expected, prediction):.4f} | "
              f"validation score: {f1_score(expected, prediction):.4f} | "
              f"validation loss: {valid_loss:.4f}\n")


def evaluate():
    with torch.no_grad():
        all_targets = []
        all_preds = []
        loss = 0
        for b in valid_buck:
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
