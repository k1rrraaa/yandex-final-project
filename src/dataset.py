import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Mean: tensor([0.4638, 0.4522, 0.4148]) (80% данных)
# Std: tensor([0.2222, 0.2198, 0.2176])

class HumanPosesDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, preload=False, label_mappings=None):
        assert mode in ['train', 'test'], "mode must be 'train' or 'test'"
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.preload = preload

        if mode == 'train':
            self.data_df = pd.read_csv(os.path.join(root_dir, 'train_answers.csv'))
            self.img_dir = os.path.join(root_dir, 'img_train')
            self.image_ids = self.data_df['img_id'].values
            self.labels = self.data_df['target_feature'].values

            # Создание маппинга: original_label -> index
            self.unique_labels = sorted(self.data_df['target_feature'].unique())
            self.label2index = {label: idx for idx, label in enumerate(self.unique_labels)}
            self.index2label = {idx: label for label, idx in self.label2index.items()}

            # Перекодировка меток
            self.labels = [self.label2index[label] for label in self.labels]

        else:  # mode == 'test'
            self.data_df = pd.read_csv(os.path.join(root_dir, 'test_dummy.csv'))
            self.img_dir = os.path.join(root_dir, 'img_test')
            self.image_ids = self.data_df['id'].values
            self.labels = None
            
            # Используем переданные маппинги или создаем пустые
            if label_mappings is not None:
                self.label2index = label_mappings['label2index']
                self.index2label = label_mappings['index2label']
            else:
                self.label2index = {}
                self.index2label = {}

        self.preloaded_images = None
        if preload:
            print(f"Preloading {len(self.image_ids)} images into RAM...")
            self.preloaded_images = [
                Image.open(os.path.join(self.img_dir, f"{img_id}.jpg")).convert("RGB")
                for img_id in self.image_ids
            ]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if self.preload and self.preloaded_images is not None:
            image = self.preloaded_images[idx]
        else:
            img_path = os.path.join(self.img_dir, f"{self.image_ids[idx]}.jpg")
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            label = self.labels[idx]
            return image, label
        else:
            return image, self.image_ids[idx]  # test возвращает id
