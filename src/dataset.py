import os
from PIL import Image
from torch.utils.data import Dataset

class HumanPosesDataset(Dataset):
    def __init__(self, data_df, img_dir, transform=None, preload=False, mode='train'):
        self.data_df = data_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.preload = preload
        self.mode = mode

        self.image_ids = self.data_df['img_id'].values

        if self.mode != 'test':
            unique_labels = sorted(self.data_df['target_feature'].unique())
            self.class_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            self.index_to_class = {v: k for k, v in self.class_to_index.items()}
            self.labels = self.data_df['target_feature'].map(self.class_to_index).values
        else:
            self.class_to_index = None
            self.index_to_class = None
            self.labels = None

        if self.preload:
            print(f"Preloading {len(self.image_ids)} images into RAM...")
            self.preloaded_images = [
                Image.open(os.path.join(self.img_dir, f"{img_id}.jpg")).convert("RGB")
                for img_id in self.image_ids
            ]
        else:
            self.preloaded_images = None

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

        if self.mode == 'test':
            return image, self.image_ids[idx]
        else:
            label = int(self.labels[idx])
            return image, label
