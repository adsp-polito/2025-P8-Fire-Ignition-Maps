import os
from typing import Tuple

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead

# Updated constants (kept identical to original for compatibility)
N_BANDS_SENTINEL = 12
N_BANDS_LANDSAT_ACTUAL = 15
N_BANDS_LANDSAT_WITH_FLAG = N_BANDS_LANDSAT_ACTUAL + 1

# DEM and streets have separate encoders in the full multimodal design
N_BANDS_DEM = 1
N_BANDS_STREETS = 1
N_BANDS_ERA5_RASTER = 2
N_BANDS_ERA5_TABULAR = 1
N_BANDS_IGNITION = 1


class MultiTaskFPN(smp.FPN):
    """
    Multi-task extension of SMP's FPN:
    - One head for burned-area segmentation (binary or multi-class)
    - One head for landcover segmentation
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights=None,
        in_channels: int = 3,
        burned_area_classes: int = 1,
        landcover_classes: int = 12,
        **kwargs,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=burned_area_classes,
            **kwargs,
        )

        self.burned_area_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=burned_area_classes,
            activation=None,
            upsampling=4,
            kernel_size=1,
        )

        self.landcover_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=landcover_classes,
            activation=None,
            upsampling=4,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass:
          x: tensor of shape (B, C, H, W)

        Returns:
          burned_area_mask, landcover_mask
        """
        features = self.encoder(x)          # list/tuple of feature maps
        decoder_output = self.decoder(features)  # <-- key change here

        burned_area_mask = self.burned_area_head(decoder_output)
        landcover_mask = self.landcover_head(decoder_output)

        return burned_area_mask, landcover_mask

class AdvancedCrossModalFusionBlock(nn.Module):
    """
    Advanced fusion block for multi-modal feature maps.

    This class is kept for future multimodal extensions.
    The current Sentinel-only baseline does not use it,
    but it remains available as a building block.
    """

    def __init__(self, channel_dims: list, output_channels: int, reduction_ratio: int = 16):
        super().__init__()

        self.modalities = len(channel_dims)
        self.channel_dims = channel_dims
        total_channels = sum(channel_dims)

        # Channel attention to weight each channel
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, max(total_channels // reduction_ratio, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(total_channels // reduction_ratio, 1), total_channels, 1),
            nn.Sigmoid(),
        )

        # Spatial attention to weight each spatial location
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(total_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Final convolution to mix modalities into a common representation
        self.conv = nn.Conv2d(total_channels, output_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, modality_features: list) -> torch.Tensor:
        """
        Args:
            modality_features: list of feature maps [f1, f2, ..., fM],
                               each of shape (B, C_m, H, W)

        Returns:
            fused: tensor of shape (B, output_channels, H, W)
        """
        # Concatenate features along channel dimension
        concat_features = torch.cat(modality_features, dim=1)

        # Channel attention
        channel_weights = self.channel_attention(concat_features)
        # Spatial attention
        spatial_weights = self.spatial_attention(concat_features)

        # Apply both attentions
        attended = concat_features * channel_weights * spatial_weights

        # Final fusion conv
        fused = self.relu(self.bn(self.conv(attended)))
        return fused


class MultiModalFPN(nn.Module):
    """
    High-level multimodal model wrapper with multi-scale fusion.

    This class extends the Sentinel-only baseline by introducing learnable
    projection layers for each additional modality (Landsat, other_data,
    ERA5 raster and tabular) at every encoder scale. An
    `AdvancedCrossModalFusionBlock` is applied at each scale to combine
    the Sentinel features with projected auxiliary features. The fused
    feature maps are then passed to the FPN decoder to produce burned
    area and landcover predictions.  The forward signature is kept
    compatible with the original design to support legacy training code.
    """

    def __init__(
        self,
        in_channels_sentinel: int = N_BANDS_SENTINEL,
        in_channels_landsat: int = N_BANDS_LANDSAT_WITH_FLAG,
        in_channels_other_data: int = N_BANDS_DEM + N_BANDS_STREETS + N_BANDS_IGNITION,
        in_channels_era5_raster: int = N_BANDS_ERA5_RASTER,
        in_channels_era5_tabular: int = N_BANDS_ERA5_TABULAR,
        in_channels_ignition_map: int = N_BANDS_IGNITION,
        num_classes: int = 1,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        landcover_classes: int = 12,
    ):
        super().__init__()

        # Base multi-task FPN using Sentinel bands as input.  This handles
        # the encoder/decoder architecture and the segmentation heads.
        self.base_fpn = MultiTaskFPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels_sentinel,
            burned_area_classes=num_classes,
            landcover_classes=landcover_classes,
        )

        # Expose encoder/decoder and heads for reuse
        self.encoder = self.base_fpn.encoder
        self.decoder = self.base_fpn.decoder
        self.ba_segmentation_head = self.base_fpn.burned_area_head
        self.lc_segmentation_head = self.base_fpn.landcover_head

        # Store channel configs for each modality
        self.in_channels_sentinel = in_channels_sentinel
        self.in_channels_landsat = in_channels_landsat
        self.in_channels_other_data = in_channels_other_data
        self.in_channels_era5_raster = in_channels_era5_raster
        self.in_channels_era5_tabular = in_channels_era5_tabular
        self.in_channels_ignition_map = in_channels_ignition_map

        # Determine channel dimensions of encoder outputs.  This list
        # corresponds to the number of channels at each scale of the
        # Sentinel encoder (e.g. [64, 256, 512, ...]).  We will fuse
        # modalities at every scale.
        sentinel_channels = list(self.encoder.out_channels)

        # Projection layers for Landsat (shared across scales)
        self.landsat_proj = nn.ModuleList(
            [nn.Conv2d(self.in_channels_landsat, c, kernel_size=1) for c in sentinel_channels]
        )

        # Projection layers for other_data (DEM + streets + ignition)
        self.other_proj = nn.ModuleList(
            [nn.Conv2d(self.in_channels_other_data, c, kernel_size=1) for c in sentinel_channels]
        )

        # Projection layers for ERA5 raster
        self.era5_raster_proj = nn.ModuleList(
            [nn.Conv2d(self.in_channels_era5_raster, c, kernel_size=1) for c in sentinel_channels]
        )

        # Projection layers for ERA5 tabular (a scalar per sample)
        self.era5_tabular_proj = nn.ModuleList(
            [nn.Linear(self.in_channels_era5_tabular, c) for c in sentinel_channels]
        )

        # Fusion blocks: one per scale.  Each block fuses the Sentinel
        # feature map with the projected Landsat, other_data, ERA5 raster and
        # ERA5 tabular features.  The output channels equal the Sentinel
        # feature channels at that scale.
        self.fusion_blocks = nn.ModuleList(
            [
                AdvancedCrossModalFusionBlock(
                    channel_dims=[c] * 5,  # 5 modalities: Sentinel, Landsat, Other, ERA5 raster, ERA5 tabular
                    output_channels=c,
                )
                for c in sentinel_channels
            ]
        )

    def forward(
        self,
        sentinel_image: torch.Tensor,
        landsat_image: torch.Tensor,
        other_data: torch.Tensor,
        ignition_map: torch.Tensor,
        era5_raster: torch.Tensor,
        era5_tabular: torch.Tensor,
    ):
        """
        Perform forward pass with multi-scale multimodal fusion.

        Args:
            sentinel_image: Tensor of shape (B, C_sentinel, H, W)
            landsat_image: Tensor of shape (B, C_landsat, H, W)
            other_data: Tensor of shape (B, C_other_data, H, W)
            ignition_map: Unused (kept for compatibility with legacy API)
            era5_raster: Tensor of shape (B, C_era5_raster, H, W)
            era5_tabular: Tensor of shape (B, C_era5_tabular)

        Returns:
            burned_area_mask: (B, num_classes, H, W)
            landcover_mask: (B, landcover_classes, H, W)
        """
        # Extract Sentinel encoder features at multiple scales
        sentinel_features = self.encoder(sentinel_image)

        fused_features = []
        # Iterate over each scale and fuse modalities
        for idx, feat in enumerate(sentinel_features):
            # Determine spatial resolution for this scale
            h, w = feat.shape[2], feat.shape[3]

            # Project Landsat and downsample to match this scale
            landsat_proj = self.landsat_proj[idx](landsat_image)
            landsat_feat = torch.nn.functional.adaptive_avg_pool2d(landsat_proj, (h, w))

            # Project other_data (DEM + streets + ignition) and downsample
            other_proj = self.other_proj[idx](other_data)
            other_feat = torch.nn.functional.adaptive_avg_pool2d(other_proj, (h, w))

            # Project ERA5 raster and downsample
            era5_proj = self.era5_raster_proj[idx](era5_raster)
            era5_feat = torch.nn.functional.adaptive_avg_pool2d(era5_proj, (h, w))

            # Project ERA5 tabular (scalar per sample) and broadcast spatially
            tab_proj = self.era5_tabular_proj[idx](era5_tabular)
            tab_feat = tab_proj.unsqueeze(-1).unsqueeze(-1)
            tab_feat = tab_feat.expand(-1, -1, h, w)

            # Fuse all modalities at this scale
            fused = self.fusion_blocks[idx]([
                feat,
                landsat_feat,
                other_feat,
                era5_feat,
                tab_feat,
            ])
            fused_features.append(fused)

        # Decode fused features with FPN decoder
        decoder_output = self.decoder(fused_features)
        burned_area_mask = self.ba_segmentation_head(decoder_output)
        landcover_mask = self.lc_segmentation_head(decoder_output)
        return burned_area_mask, landcover_mask