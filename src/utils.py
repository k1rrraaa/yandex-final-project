import random
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import clear_output
import time
import webbrowser
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score, recall_score
import pandas as pd
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_losses(train_losses, val_losses, train_f1s, val_f1s, show=True, save_path=None, clear=True, serve_locally=False):
    if clear:
        clear_output(wait=True)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('Loss', 'F1 Score')
    )

    fig.add_trace(
        go.Scatter(y=train_losses, name='train loss', line=dict(color='blue', width=2), mode='lines+markers'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=val_losses, name='val loss', line=dict(color='red', width=2), mode='lines+markers'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(y=train_f1s, name='train f1', line=dict(color='blue', width=2), mode='lines+markers'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=val_f1s, name='val f1', line=dict(color='red', width=2), mode='lines+markers'),
        row=1, col=2
    )

    fig.update_layout(
        width=1100,
        height=450,
        title_text="Training Metrics",
        template='plotly_white',
        showlegend=True
    )

    if save_path:
        fig.write_image(save_path)

    if serve_locally:
        html_path = "training_plot.html"
        fig.write_html(html_path)
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
    elif show:
        fig.show()
        time.sleep(0.3)


def compute_f1(y_true, y_pred, average='macro'):
    return f1_score(y_true, y_pred, average=average)


def show_classification_report(y_true, y_pred, label_names=None):
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))


def show_confusion_matrix(y_true, y_pred, label_names=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)

    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 15})

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def evaluate_model(model, dataloader, device, label_names=None):
    if label_names is None and hasattr(dataloader.dataset, 'index_to_class'):
        index_to_class = dataloader.dataset.index_to_class
        label_names = [index_to_class[i] for i in range(len(index_to_class))]

    if label_names is not None:
        label_names = [str(label) for label in label_names]

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print("\nClassification Report:")
    show_classification_report(y_true, y_pred, label_names)

    print("\nConfusion Matrix:")
    show_confusion_matrix(y_true, y_pred, label_names)

    print(f"\nMacro F1-score: {f1:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")

    return {'f1': f1, 'precision': precision, 'recall': recall}


def load_best_model(model, checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"✅ Loaded model weights from {checkpoint_path}")
    return model


def load_training_state(model, optimizer, scheduler, checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    model.to(device)
    print(f"✅ Loaded training state from {checkpoint_path}")
    return model, optimizer, scheduler, checkpoint['epoch'] + 1, checkpoint['history'], checkpoint['best_f1']


class MixupCutMixAugmenter:
    def __init__(self, alpha=1.0, p_mixup=0.5):
        self.alpha = alpha
        self.p_mixup = p_mixup

    def __call__(self, x, y):
        if random.random() < self.p_mixup:
            return self.mixup(x, y)
        else:
            return self.cutmix(x, y)

    def mixup(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, (y_a, y_b, lam)

    def cutmix(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size, _, height, width = x.size()
        index = torch.randperm(batch_size).to(x.device)

        cut_rat = np.sqrt(1. - lam)
        cut_w = int(width * cut_rat)
        cut_h = int(height * cut_rat)

        cx = random.randint(0, width)
        cy = random.randint(0, height)

        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)

        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        y_a, y_b = y, y[index]

        return x, (y_a, y_b, lam)


class ResizeWithAspectRatioPadding:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL.Image, got {type(img)}")

        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        img = img.resize((new_w, new_h), Image.BILINEAR)

        pad_w = self.size - new_w
        pad_h = self.size - new_h

        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        img = ImageOps.expand(img, border=padding, fill=self.fill)

        return img


@torch.no_grad()
def make_submission(model, test_loader, device, index_to_class, output_path="submission.csv"):
    model.eval()
    model.to(device)

    all_preds = []
    all_ids = []

    # Получаем index2label из оригинального датасета, даже если используется Subset
    dataset = test_loader.dataset
    if hasattr(dataset, 'dataset'):  # если используется Subset
        dataset = dataset.dataset
    index2label = getattr(dataset, 'index2label', None)

    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_ids.extend(ids)

    all_ids = [int(i) for i in all_ids]
    decoded_preds = [index_to_class[int(p)] for p in all_preds]

    if index2label is not None:
        all_preds = [index2label[p] for p in all_preds]

    df = pd.DataFrame({
        'id': all_ids,
        'target_feature': decoded_preds
    })

    df = df.sort_values("id")
    df.to_csv(output_path, index=False)
    print(f"✅ Submission saved to: {output_path}")

