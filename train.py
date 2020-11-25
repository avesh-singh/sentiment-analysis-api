from model import model, nn
from data import load_data
from sklearn.metrics import precision_score, recall_score
from constants import *
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np

train_loader, valid_loader, test_loader, train_len, valid_len, test_len = load_data()
model = model.to(device)
criterion = nn.CrossEntropyLoss()

# as per original bert paper, fine-tuning is done by Adam optimizer with weight decay
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


def train(model):
    model = model.train()
    optimizer.zero_grad()

    losses = []
    accurate_preds = 0
    all_targets = []
    all_predictions = []
    for d in train_loader:
        inputs = d['input_ids'].to(device)
        masks = d['attention_mask'].to(device)
        all_targets.extend(list(d['targets'].squeeze().numpy()))
        targets = d['targets'].to(device)
        outputs = model(
            input_ids=inputs,
            attention_mask=masks
        )
        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, targets)
        all_predictions.extend(list(preds.cpu().squeeze().numpy()))
        accurate_preds += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    return accurate_preds / train_len, np.mean(losses), all_targets, all_predictions


def train_iters():
    best_valid_acc = 0
    for e in range(EPOCHS):
        print(f"epoch: {e}")
        train_acc, train_loss, train_tgt, train_pred = train(model)
        print(f"training loss: {train_loss:.4f} | training accuracy: {train_acc:.4f} | training precision:"
              f" {precision_score(train_tgt, train_pred):.4f} | training recall:"
              f" {recall_score(train_tgt, train_pred):.4f}")
        valid_acc, valid_loss, expected, prediction = evaluate(model, valid_loader, valid_len)
        print(f"validation loss: {valid_loss:.4f} | validation accuracy: {valid_acc:.4f} | validation precision:"
              f" {precision_score(expected, prediction):.4f} | validation recall: "
              f"{recall_score(expected, prediction):.4f}")
        if best_valid_acc < valid_acc:
            print("new best model! improvement: %f" % (best_valid_acc - valid_acc))
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), 'model.pt')


def evaluate(model, loader, data_len):
    model = model.eval()
    with torch.no_grad():
        all_targets = []
        all_preds = []
        accurate_preds = 0
        losses = []
        for d in loader:
            inputs = d['input_ids'].to(device)
            masks = d['attention_mask'].to(device)
            all_targets.extend(list(d['targets'].squeeze().numpy()))
            targets = d['targets'].to(device)
            outputs = model(
                input_ids=inputs,
                attention_mask=masks
            )
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, targets)
            all_preds.extend(list(preds.cpu().squeeze().numpy()))
            accurate_preds += torch.sum(preds == targets)
            losses.append(loss.item())
    return accurate_preds / data_len, np.mean(losses), all_targets, all_preds


if __name__ == '__main__':
    train_iters()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    acc, loss, tgt, pred = evaluate(model, test_loader, test_len)
    print(f"test loss: {loss:.4f} | test accuracy: {acc:.4f} | test precision:"
          f" {precision_score(tgt, pred):.4f} | test recall: "
          f"{recall_score(tgt, pred):.4f}")
