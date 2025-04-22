from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class MAEDataset(Dataset):
    def __init__(self, root_dir, img_size=224, transform=None):
        self.root_dir = root_dir
        self.img_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return self.transform(img)
