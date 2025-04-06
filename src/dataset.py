import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class HumanPosesDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, preload=False):
        assert mode in ['train', 'test'], "mode must be 'train' or 'test'"
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.preload = preload

        if mode == 'train':
            self.data_df = pd.read_csv(os.path.join(root_dir, 'train_answers.csv'))
            self.img_dir = os.path.join(root_dir, 'img_train')
        else:
            self.data_df = pd.read_csv(os.path.join(root_dir, 'test_dummy.csv'))
            self.img_dir = os.path.join(root_dir, 'img_test')

        if mode == 'train':
            self.image_ids = self.data_df['img_id'].values
        else:
            self.image_ids = self.data_df['id'].values

        self.labels = self.data_df['target_feature'].values if mode == 'train' else None

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

