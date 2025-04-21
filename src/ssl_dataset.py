from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import os

class SimCLRTransform:
    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor()
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class SimCLRDataset(Dataset):
    def __init__(self, image_folder):
        self.paths = [os.path.join(image_folder, f)
                      for f in os.listdir(image_folder)
                      if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.loader = default_loader
        self.transform = SimCLRTransform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = self.loader(self.paths[idx])
        x1, x2 = self.transform(image)
        return x1, x2
