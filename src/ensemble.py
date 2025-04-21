from typing import Dict, List, Optional, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import f1_score

class TestDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        ids: List[str],
        transforms: Optional[Callable] = None,
        preload_to_ram: bool = False
    ):
        self.transforms = transforms
        self.ids = ids
        self.preload_to_ram = preload_to_ram

        if preload_to_ram:
            self.images = [Image.open(p).convert("RGB") for p in image_paths]
        else:
            self.image_paths = image_paths

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self.images[idx] if self.preload_to_ram else Image.open(self.image_paths[idx]).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, self.ids[idx]


class WeightedEnsemble:
    def __init__(self, model_weights: Dict[str, float]):
        total = sum(model_weights.values())
        assert total > 0, "Сумма весов должна быть положительной"
        self.model_weights = {k: v / total for k, v in model_weights.items()}

    def predict_proba(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        for name in self.model_weights:
            assert name in predictions, f"Нет предсказаний от модели '{name}'"

        weighted_sum = None
        for name, weight in self.model_weights.items():
            probs = predictions[name]
            if isinstance(probs, torch.Tensor):
                probs = probs.detach().cpu().numpy()
            if weighted_sum is None:
                weighted_sum = weight * probs
            else:
                weighted_sum += weight * probs
        return weighted_sum

    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        probs = self.predict_proba(predictions)
        return np.argmax(probs, axis=1)

    def make_submission(
        self,
        predictions: Dict[str, np.ndarray],
        image_ids: List[str],
        index_to_class: Dict[int, str],
        output_path: str = "submission.csv"
    ) -> None:
        preds = self.predict(predictions)
        decoded_preds = [index_to_class[int(p)] for p in preds]
        all_ids = [int(i) for i in image_ids]

        df = pd.DataFrame({
            'id': all_ids,
            'target_feature': decoded_preds
        })
        df = df.sort_values("id")
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Submission saved to: {output_path}")

    @torch.no_grad()
    def predict_from_models(
        self,
        models: Dict[str, torch.nn.Module],
        dataloader: DataLoader,
        device: torch.device,
        tta_transforms: Optional[List[Callable]] = None
    ) -> Dict[str, np.ndarray]:

        predictions: Dict[str, List[np.ndarray]] = {name: [] for name in models.keys()}
        ids_all: List[str] = []

        tta_list = tta_transforms if tta_transforms else [lambda x: x]
        tta_n = len(tta_list)

        for images, ids in tqdm(dataloader, desc="Predicting with ensemble"):
            ids_all.extend(ids)

            for name, model in models.items():
                model.eval()
                model.to(device)
                votes = None

                for transform in tta_list:
                    batch = torch.stack([transform(img) for img in images]).to(device)
                    logits = model(batch).detach().cpu().numpy()

                    if votes is None:
                        votes = logits
                    else:
                        votes += logits

                votes /= tta_n
                predictions[name].extend(votes)

        # в np.ndarray
        for name in predictions:
            predictions[name] = np.stack(predictions[name])

        return predictions, ids_all

    def validate(
        self,
        predictions: Dict[str, np.ndarray],
        true_labels: List[int],
        average: str = "macro"
    ) -> float:

        preds = self.predict(predictions)
        score = f1_score(true_labels, preds, average=average)
        print(f"✅ Validation F1-score ({average}): {score:.4f}")
        return score
