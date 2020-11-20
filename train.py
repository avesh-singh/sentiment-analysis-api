from model import *
from data import prepare_data, device, prepare_data_keras
from sklearn.metrics import confusion_matrix, precision_score, recall_score

EMBEDDING_SIZE = 256
HIDDEN_SIZE = 64
BATCH_SIZE = 32
DROPOUT = 0.2
CLIP = 1
EPOCHS = 15

# train_buck, valid_buck, test_buck, text_field, label_field = prepare_data(BATCH_SIZE)
train_buck, valid_buck, test_buck, vocab_size = prepare_data_keras(BATCH_SIZE)
model = Model(vocab_size, EMBEDDING_SIZE, HIDDEN_SIZE, 1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())


def train(batch):
    model.zero_grad()
    label = batch[1].view(-1, 1).to(dtype=torch.float)
    text = batch[0]
    output = model(text)
    loss = criterion(output, label)
    loss.backward()
    # nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    optimizer.step()
    return loss.item() / len(batch), list(label.squeeze().cpu().numpy()), list(map(int, output > 0))


def train_iters():
    best_valid_loss = float('inf')
    for e in range(EPOCHS):
        train_loss = 0
        train_tgt = []
        train_pred = []
        exp = []
        pred = []
        for i, batch in enumerate(train_buck):
            loss, train_target, train_prediction = train(batch)
            train_loss += loss / len(batch)
            train_tgt.extend(train_target)
            train_pred.extend(train_prediction)

        valid_loss, expected, prediction = evaluate()
        if valid_loss < best_valid_loss:
            print("new best model! improvement: %f" % (best_valid_loss - valid_loss))
            best_valid_loss = valid_loss
            torch.save(model.state_dict, open('model.pt', 'wb'))
        exp.extend(expected)
        pred.extend(prediction)
        print(f"epochs: {e} | training precision: {precision_score(train_tgt, train_pred):.4f}   | training recall:"
              f" {recall_score(train_tgt, train_pred):.4f}   |  training loss: {train_loss / len(train_buck):.4f}")
        print(f"epochs: {e} | validation precision: {precision_score(expected, prediction):.4f} | validation recall: "
              f"{recall_score(expected, prediction):.4f} | validation loss: {valid_loss:.4f}\n")


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
            loss += criterion(output, label) / len(b)
            pred = list(map(int, output > 0))
            all_preds.extend(pred)
    return loss / len(valid_buck), all_targets, all_preds


train_iters()