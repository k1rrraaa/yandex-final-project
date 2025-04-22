from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm


class TestDataset(Dataset):
    def __init__(self, image_paths, ids):
        self.image_paths = image_paths
        self.ids = ids

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return img, self.ids[idx]


mean = [0.4638, 0.4522, 0.4148]
std = [0.2222, 0.2198, 0.2176]

tta_transforms = [
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
    transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]),
]


def apply_class_prior_correction(logits_tensor, train_labels):

    class_counts = np.bincount(train_labels, minlength=logits_tensor.shape[1])
    priors = class_counts / class_counts.sum()
    log_priors = torch.log(torch.tensor(priors + 1e-6, dtype=torch.float32)).to(logits_tensor.device)

    corrected_logits = logits_tensor - log_priors.unsqueeze(0)
    probs = torch.softmax(corrected_logits, dim=1)
    return probs


@torch.no_grad()
def make_submission_with_tta(
    model,
    test_loader,
    device,
    index_to_class,
    output_path="submission.csv",
    use_class_prior_correction=False,
    train_labels=None
):
    model.eval()
    model.to(device)

    NUM_CLASSES = len(index_to_class)
    all_preds = []
    all_ids = []

    for images, ids in tqdm(test_loader, desc="Predicting with TTA"):
        batch_size = len(images)
        votes = torch.zeros((batch_size, NUM_CLASSES), device=device)

        for transform in tta_transforms:
            tta_batch = torch.stack([transform(img) for img in images]).to(device)
            logits = model(tta_batch)
            logits = logits[:, :NUM_CLASSES]
            votes += logits

        if use_class_prior_correction:
            if train_labels is None:
                raise ValueError("train_labels must be provided when using class prior correction.")
            probs = apply_class_prior_correction(votes, train_labels)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
        else:
            preds = torch.argmax(votes, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_ids.extend(ids)

    all_ids = [int(i) for i in all_ids]
    decoded_preds = [index_to_class[int(p)] for p in all_preds]

    df = pd.DataFrame({
        'id': all_ids,
        'target_feature': decoded_preds
    })
    df = df.sort_values("id")
    df.to_csv(output_path, index=False)
    print(f"✅ Submission with TTA saved to: {output_path}")
