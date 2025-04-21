import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import RandAugment, RandomErasing
from PIL import Image
import cv2
import torch


class StrongAugment:

    def __init__(
        self,
        image_size: int = 224,
        use_randaug: bool = True,
        rand_n: int = 2,
        rand_m: int = 9,
        use_gridmask: bool = False,
        use_random_erasing: bool = True,
    ):
        self.randaug = RandAugment(num_ops=rand_n, magnitude=rand_m) if use_randaug else None

        self.eraser = (
            RandomErasing(
                p=0.25,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value='random',
            )
            if use_random_erasing
            else None
        )

        self.albu = A.Compose(
            [
                A.LongestMaxSize(max_size=image_size + 32, interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(min_height=image_size + 32, min_width=image_size + 32,
                              border_mode=cv2.BORDER_CONSTANT,
                              fill=0,
                              p=1.0),

                A.RandomResizedCrop(size=(image_size, image_size),
                                    scale=(0.65, 1.0),
                                    ratio=(0.75, 1.33), p=1.0),
                A.HorizontalFlip(p=0.5),

                A.OneOf(
                    [
                        A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0)
                        if use_gridmask
                        else A.GridDropout(ratio=0.5,
                              holes_number_xy=(4, 4),
                              random_offset=True,
                              fill=0, p=1.0),
                        A.CoarseDropout(num_holes_range=(4, 8),
                                        hole_height_range=(16, 32),
                                        hole_width_range=(16, 32),
                                        fill=0, p=1.0),
                    ],
                    p=0.5,
                ),

                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5, p=1.0),
                        A.MedianBlur(blur_limit=3, p=1.0),
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    ],
                    p=0.2,
                ),

                A.OneOf(
                    [
                        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
                        A.HueSaturationValue(
                            hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0
                        ),
                        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
                        A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0),
                    ],
                    p=0.4,
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.2),

                A.Downscale(
                    scale_range=(0.6, 0.9),
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LINEAR,
                    },
                    p=0.2
                ),
                A.ImageCompression(quality_range=(40, 90), p=0.2),

                A.Normalize(
                    mean=(0.4638, 0.4522, 0.4148),
                    std=(0.2222, 0.2198, 0.2176),
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        if self.randaug is not None:
            img = self.randaug(img)

        tensor = self.albu(image=np.asarray(img))["image"]

        if self.eraser is not None:
            tensor = self.eraser(tensor)

        return tensor

class TestAugment:
    def __init__(self, image_size: int = 224):
        self.albu = A.Compose(
            [
                A.LongestMaxSize(max_size=image_size + 32, interpolation=cv2.INTER_LINEAR),
                A.PadIfNeeded(min_height=image_size + 32, min_width=image_size + 32,
                              border_mode=cv2.BORDER_CONSTANT,
                              fill=0,
                              p=1.0),
                A.CenterCrop(height=image_size, width=image_size),
                A.Normalize(
                    mean=(0.4638, 0.4522, 0.4148),
                    std=(0.2222, 0.2198, 0.2176),
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.albu(image=np.asarray(img))["image"]
