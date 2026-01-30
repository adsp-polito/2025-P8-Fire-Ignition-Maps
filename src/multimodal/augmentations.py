import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from typing import Tuple, Dict
import cv2

N_BANDS_SENTINEL = 12
N_BANDS_LANDSAT_ACTUAL = 15
N_BANDS_LANDSAT_WITH_FLAG = N_BANDS_LANDSAT_ACTUAL + 1


class SentinelAugmentations:
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        p_flip: float = 0.6,
        p_color: float = 0.3,
    ):
        self.target_size = target_size

        self.spatial_transform = A.Compose(
            [
                A.HorizontalFlip(p=p_flip),
                A.VerticalFlip(p=p_flip),
                A.RandomRotate90(p=p_flip),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.10,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5,
                ),
                A.Resize(height=self.target_size[0], width=self.target_size[1]),
            ],
            additional_targets={
                "landsat_image": "image",
                "streets_image": "image",
                "dem_image": "image",
                "ignition_pt": "mask",
                "era5_image": "image",
                "landcover_np": "image",
                "mask": "mask",
            },
        )

        # Color transforms ONLY on Sentinel
        self.color_transform = A.Compose(
            [
                A.RandomBrightnessContrast(p=p_color),
                A.RandomGamma(p=p_color),
                A.GaussNoise(var_limit=(10.0, 50.0), p=p_color),
            ]
        )

        self.to_tensor = ToTensorV2()

    @staticmethod
    def _ensure_hwc(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img[..., np.newaxis]
        return img

    @staticmethod
    def _to_chw_tensor(img: np.ndarray) -> torch.Tensor:
        if img.ndim == 2:
            img = img[..., np.newaxis]
        img = img.astype(np.float32)
        return torch.from_numpy(img).permute(2, 0, 1)

    @staticmethod
    def _to_mask_tensor(mask: np.ndarray) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0)

    def __call__(
        self,
        image: np.ndarray,
        landsat_image: np.ndarray,
        streets_image: np.ndarray,
        dem_image: np.ndarray,
        ignition_pt: np.ndarray,
        era5_image: np.ndarray,
        landcover_np: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, torch.Tensor]:

        image = self._ensure_hwc(image)
        landsat_image = self._ensure_hwc(landsat_image)
        streets_image = self._ensure_hwc(streets_image)
        dem_image = self._ensure_hwc(dem_image)
        ignition_pt = self._ensure_hwc(ignition_pt)
        era5_image = self._ensure_hwc(era5_image)
        landcover_np = self._ensure_hwc(landcover_np)
        if mask.ndim == 3:
            mask = mask[..., 0]

        augmented_spatial = self.spatial_transform(
            image=image,
            landsat_image=landsat_image,
            streets_image=streets_image,
            dem_image=dem_image,
            ignition_pt=ignition_pt,
            era5_image=era5_image,
            landcover_np=landcover_np,
            mask=mask,
        )

        sentinel_spatial = augmented_spatial["image"]
        landsat_spatial = augmented_spatial["landsat_image"]
        streets_spatial = augmented_spatial["streets_image"]
        dem_spatial = augmented_spatial["dem_image"]
        ignition_spatial = augmented_spatial["ignition_pt"]
        era5_spatial = augmented_spatial["era5_image"]
        landcover_spatial = augmented_spatial["landcover_np"]
        mask_spatial = augmented_spatial["mask"]

        # Color on Sentinel only
        color_aug = self.color_transform(image=sentinel_spatial)
        sentinel_color = color_aug["image"]
        landsat_color = landsat_spatial  # unchanged

        streets_final = streets_spatial
        dem_final = dem_spatial
        ignition_final = ignition_spatial
        era5_final = era5_spatial
        landcover_final = landcover_spatial
        mask_final = mask_spatial

        sentinel_image_tensor = self._to_chw_tensor(sentinel_color)
        landsat_image_tensor = self._to_chw_tensor(landsat_color)
        streets_tensor = self._to_chw_tensor(streets_final)
        dem_tensor = self._to_chw_tensor(dem_final)
        ignition_pt_tensor = self._to_chw_tensor(ignition_final)
        era5_tensor = self._to_chw_tensor(era5_final)
        landcover_tensor = self._to_chw_tensor(landcover_final)
        mask_tensor = self._to_mask_tensor(mask_final)

        return {
            "sentinel_image": sentinel_image_tensor,
            "landsat_image": landsat_image_tensor,
            "streets_image": streets_tensor,
            "dem_image": dem_tensor,
            "ignition_pt": ignition_pt_tensor,
            "era5_image": era5_tensor,
            "landcover_np": landcover_tensor,
            "mask": mask_tensor,
        }