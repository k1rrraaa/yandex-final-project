from tqdm import tqdm
from sklearn.metrics import f1_score
import torch

def training_epoch(model, optimizer, criterion, train_loader, device, tqdm_desc):
    model.train()
    train_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(train_loader, desc=tqdm_desc):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.shape[0]
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    train_loss /= len(train_loader.dataset)
    train_f1 = f1_score(torch.cat(all_labels), torch.cat(all_preds), average='macro')
    return train_loss, train_f1


@torch.no_grad()
def validation_epoch(model, criterion, val_loader, device, tqdm_desc):
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(val_loader, desc=tqdm_desc):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        val_loss += loss.item() * images.shape[0]
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    val_loss /= len(val_loader.dataset)
    val_f1 = f1_score(torch.cat(all_labels), torch.cat(all_preds), average='macro')
    return val_loss, val_f1


def train_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs, save_path=None, plot_fn=None):
    train_losses, val_losses, train_f1s, val_f1s = [], [], [], []
    best_f1 = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_f1 = training_epoch(model, optimizer, criterion, train_loader, device, f"Train {epoch}")
        val_loss, val_f1 = validation_epoch(model, criterion, val_loader, device, f"Val {epoch}")

        if scheduler is not None:
            scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            if save_path is not None:
                torch.save(model.state_dict(), save_path)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        if plot_fn:
            plot_fn(train_losses, val_losses, train_f1s, val_f1s)

    print(f"Training completed. Best Val F1: {best_f1:.4f}")
    return train_losses, val_losses, train_f1s, val_f1s
