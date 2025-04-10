from tqdm import tqdm
from sklearn.metrics import f1_score
import torch
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast



def training_epoch(model, optimizer, criterion, train_loader, device, tqdm_desc, batch_augment_fn=None, scheduler=None, scaler=None):
    model.train()
    train_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(train_loader, desc=tqdm_desc):
        images, labels = images.to(device), labels.to(device)

        if batch_augment_fn is not None:
            images, labels = batch_augment_fn(images, labels)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type='cuda'):
                logits = model(images)
                if isinstance(labels, tuple) and len(labels) == 3:
                    y_a, y_b, lam = labels
                    loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                    labels_for_f1 = y_a
                else:
                    loss = criterion(logits, labels)
                    labels_for_f1 = labels

            scaler.scale(loss).backward()
            # clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            if isinstance(labels, tuple) and len(labels) == 3:
                y_a, y_b, lam = labels
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                labels_for_f1 = y_a
            else:
                loss = criterion(logits, labels)
                labels_for_f1 = labels

            loss.backward()
            # clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        train_loss += loss.item() * images.size(0)
        all_preds.append(logits.detach().argmax(dim=1).cpu())
        all_labels.append(labels_for_f1.cpu())

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

        val_loss += loss.item() * images.size(0)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    val_loss /= len(val_loader.dataset)
    val_f1 = f1_score(torch.cat(all_labels), torch.cat(all_preds), average='macro')
    return val_loss, val_f1
