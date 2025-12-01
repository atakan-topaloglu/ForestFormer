"""
PointTransformerV3 backbone wrapper for ForestFormer/OneFormer3D.

Converts between spconv.SparseConvTensor and PTV3's Point format.
"""

import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import spconv.pytorch as spconv

from mmdet3d.registry import MODELS

# Add parent of PTV3 to path so we can import as a package
# This handles the relative imports in model.py (from .serialization import encode)
PTV3_PARENT = Path(__file__).parent.parent.parent  # projects/ptv3/
if str(PTV3_PARENT) not in sys.path:
    sys.path.insert(0, str(PTV3_PARENT))

from PointTransformerV3.model import PointTransformerV3, Point, offset2batch, batch2offset


@MODELS.register_module()
class PTV3Backbone(nn.Module):
    """PointTransformerV3 backbone wrapper for ForestFormer.
    
    This wrapper handles the conversion between spconv.SparseConvTensor
    (used by ForestFormer) and PTV3's Point format.
    
    Args:
        in_channels (int): Number of input feature channels.
        order (tuple): Serialization orders for attention.
        stride (tuple): Downsampling strides for encoder stages.
        enc_depths (tuple): Number of blocks at each encoder stage.
        enc_channels (tuple): Channel dimensions at each encoder stage.
        enc_num_head (tuple): Number of attention heads at each encoder stage.
        enc_patch_size (tuple): Patch sizes for attention at each encoder stage.
        dec_depths (tuple): Number of blocks at each decoder stage.
        dec_channels (tuple): Channel dimensions at each decoder stage.
        dec_num_head (tuple): Number of attention heads at each decoder stage.
        dec_patch_size (tuple): Patch sizes for attention at each decoder stage.
        mlp_ratio (float): MLP expansion ratio.
        qkv_bias (bool): Whether to use bias in QKV projection.
        qk_scale (float): Scale factor for QK attention.
        attn_drop (float): Attention dropout rate.
        proj_drop (float): Projection dropout rate.
        drop_path (float): Drop path rate.
        shuffle_orders (bool): Whether to shuffle serialization orders.
        pre_norm (bool): Whether to apply pre-normalization.
        enable_rpe (bool): Whether to enable relative position encoding.
        enable_flash (bool): Whether to enable flash attention.
        upcast_attention (bool): Whether to upcast attention to fp32.
        upcast_softmax (bool): Whether to upcast softmax to fp32.
        pdnorm_bn (bool): Whether to use PDNorm for BatchNorm.
        pdnorm_ln (bool): Whether to use PDNorm for LayerNorm.
        pdnorm_decouple (bool): Whether to decouple PDNorm per condition.
        pdnorm_adaptive (bool): Whether to use adaptive PDNorm.
        pdnorm_affine (bool): Whether to use affine in PDNorm.
        pdnorm_conditions (tuple): Dataset conditions for PDNorm.
        return_blocks (bool): Compatibility flag (ignored, for SpConvUNet interface).
        grid_size (float): Voxel/grid size for the point cloud.
    """
    
    def __init__(
        self,
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        return_blocks=False,  # Ignored, for compatibility with SpConvUNet interface
        grid_size=0.06,
        pretrained=None,  # Path to pretrained Pointcept checkpoint
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.return_blocks = return_blocks  # Store but ignore (PTV3 doesn't use this)
        self.out_channels = dec_channels[0]  # Output feature dimension
        self.pretrained = pretrained
        
        # Build PTV3 backbone
        self.ptv3 = PointTransformerV3(
            in_channels=in_channels,
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            shuffle_orders=shuffle_orders,
            pre_norm=pre_norm,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            cls_mode=False,  # We want encoder-decoder, not classification
            pdnorm_bn=pdnorm_bn,
            pdnorm_ln=pdnorm_ln,
            pdnorm_decouple=pdnorm_decouple,
            pdnorm_adaptive=pdnorm_adaptive,
            pdnorm_affine=pdnorm_affine,
            pdnorm_conditions=pdnorm_conditions,
        )
        
        # Load pretrained weights if provided
        if self.pretrained:
            self.load_pretrained(self.pretrained)
    
    def load_pretrained(self, checkpoint_path):
        """Load pretrained Pointcept PTV3 weights.
        
        Pointcept checkpoint structure:
            - 'state_dict' or 'model': contains weights
            - Keys like 'backbone.xxx' -> we need 'ptv3.xxx'
        
        What loads:
            - Encoder blocks (enc.enc0-4): channel dims match if using original PTv3 config
            - Decoder blocks (dec.dec0-3): channel dims match if using original PTv3 config
            - All attention, MLP, norm layers in enc/dec
        
        What CAN'T load (shape mismatch):
            - Embedding/stem layer: pretrained has in_channels=6, we have 64 (from input_conv)
            - This is expected and unavoidable - embedding will be randomly initialized
        
        Args:
            checkpoint_path: Path to Pointcept checkpoint (.pth file)
        """
        import os
        print(f"[PTV3Backbone] Loading pretrained weights from {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get state dict from checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Map Pointcept keys to our wrapper keys
        # Handle various key formats from distributed/non-distributed training:
        # - 'module.backbone.enc.xxx' (DDP)
        # - 'backbone.enc.xxx' 
        # - 'enc.xxx'
        # All map to: 'ptv3.enc.xxx'
        new_state_dict = {}
        skipped_embedding = 0
        loaded_count = 0
        
        for key, value in state_dict.items():
            # Skip segmentation head and criterion
            if 'seg_head' in key or 'criterion' in key:
                continue
            
            # Skip embedding/stem layer - EXPECTED: shape mismatch (pretrained=6, ours=64 from input_conv)
            # The embedding layer will be randomly initialized and trained from scratch
            if 'embedding' in key or 'stem' in key:
                skipped_embedding += 1
                continue
            
            # Extract the backbone-relative key
            new_key = key
            
            # Remove 'module.' prefix (from DDP)
            if new_key.startswith('module.'):
                new_key = new_key[len('module.'):]
            
            # Remove 'backbone.' prefix
            if new_key.startswith('backbone.'):
                new_key = new_key[len('backbone.'):]
            
            # Add our prefix
            new_key = 'ptv3.' + new_key
            
            # Fix spconv weight format: old format (C_out, C_in, kD, kH, kW) -> new format (C_in, kD, kH, kW, C_out)
            # This affects all SubMConv3d/SparseConv3d weights which are 5D tensors
            # Permutation (1,2,3,4,0) matches the fix_spconv_checkpoint.py script
            if value.dim() == 5 and new_key.endswith('weight'):
                value = value.permute(1, 2, 3, 4, 0).contiguous()
            
            new_state_dict[new_key] = value
            loaded_count += 1
        
        print(f"[PTV3Backbone] Processed {loaded_count} enc/dec weights")
        print(f"[PTV3Backbone] Skipped {skipped_embedding} embedding weights (expected - shape mismatch)")
        
        # Load with strict=False to allow missing/extra keys
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        
        print(f"[PTV3Backbone] Loaded pretrained weights:")
        print(f"  - Missing keys: {len(missing)} (embedding layer - will train from scratch)")
        print(f"  - Unexpected keys: {len(unexpected)}")
        if missing and len(missing) <= 10:
            print(f"  - Missing: {missing}")
        elif missing:
            print(f"  - Example missing: {missing[:5]}")
        if unexpected:
            print(f"  - Example unexpected: {unexpected[:3]}")
    
    def spconv_to_point(self, x: spconv.SparseConvTensor) -> tuple:
        """Convert spconv.SparseConvTensor to PTV3 data_dict format.
        
        PTv3 expects points sorted by batch index. We sort them here and
        return the inverse permutation to restore original order after processing.
        
        Args:
            x: SparseConvTensor with:
                - features: (N, C) tensor of point features
                - indices: (N, 4) tensor of [batch_idx, x, y, z] (MinkowskiEngine format)
                - spatial_shape: [D, H, W] voxel grid dimensions
                - batch_size: number of batches
        
        Returns:
            tuple: (data_dict, sort_idx, unsort_idx)
                - data_dict: PTV3-compatible data dict with feat, coord, grid_coord, batch, offset
                - sort_idx: indices that sort points by batch
                - unsort_idx: indices that restore original order
        """
        features = x.features  # (N, C)
        indices = x.indices    # (N, 4) - [batch_idx, x, y, z] (from MinkowskiEngine)
        
        # Extract batch indices and grid coordinates
        batch_indices = indices[:, 0].long()  # (N,)
        # ForestFormer uses MinkowskiEngine which outputs [batch, x, y, z]
        # PTV3 expects [x, y, z], so just take columns 1, 2, 3
        grid_coord = indices[:, [1, 2, 3]].int()  # (N, 3) - [x, y, z]
        
        # Sort points by batch index (PTv3 expects this)
        sort_idx = torch.argsort(batch_indices, stable=True)
        unsort_idx = torch.argsort(sort_idx)  # Inverse permutation
        
        # Apply sorting
        features_sorted = features[sort_idx]
        batch_sorted = batch_indices[sort_idx]
        grid_coord_sorted = grid_coord[sort_idx]
        
        # Compute float coordinates from grid coordinates
        coord_sorted = grid_coord_sorted.float() * self.grid_size  # (N, 3)
        
        # Compute offset using batch2offset (works correctly on sorted batch indices)
        offset = batch2offset(batch_sorted)
        
        data_dict = {
            "feat": features_sorted,
            "coord": coord_sorted,
            "grid_coord": grid_coord_sorted,
            "batch": batch_sorted,
            "offset": offset,
        }
        
        return data_dict, sort_idx, unsort_idx
    
    def point_to_spconv(
        self, 
        point: Point, 
        unsort_idx: torch.Tensor,
        original_indices: torch.Tensor,
        spatial_shape: list,
        batch_size: int
    ) -> spconv.SparseConvTensor:
        """Convert PTV3 Point output back to spconv.SparseConvTensor.
        
        Args:
            point: PTV3 Point object with feat attribute
            unsort_idx: Indices to restore original point order (before batch sorting)
            original_indices: Original spconv indices (N, 4) to preserve ordering
            spatial_shape: Original spatial shape [D, H, W]
            batch_size: Batch size
            
        Returns:
            spconv.SparseConvTensor with output features in original order
        """
        features = point.feat  # (N, C_out) - in sorted order
        
        # Restore original point order
        features_unsorted = features[unsort_idx]
        
        # Create new SparseConvTensor with output features
        # Use original indices to maintain point ordering
        output = spconv.SparseConvTensor(
            features=features_unsorted,
            indices=original_indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )
        
        return output
    
    def forward(self, x: spconv.SparseConvTensor, previous_outputs=None):
        """Forward pass.
        
        Args:
            x: spconv.SparseConvTensor input
            previous_outputs: Ignored (for SpConvUNet interface compatibility)
            
        Returns:
            spconv.SparseConvTensor: Output features
            If return_blocks=True: returns (output, [output]) for compatibility
        """
        # Store original spconv info for reconstruction
        original_indices = x.indices.clone()
        spatial_shape = x.spatial_shape
        batch_size = x.batch_size
        
        # Convert to PTV3 format (sorts by batch, returns unsort indices)
        data_dict, sort_idx, unsort_idx = self.spconv_to_point(x)
        
        # Run PTV3
        point = self.ptv3(data_dict)
        
        # Convert back to spconv format (restores original order)
        output = self.point_to_spconv(
            point, unsort_idx, original_indices, spatial_shape, batch_size
        )
        
        if self.return_blocks:
            # Return format compatible with SpConvUNet when return_blocks=True
            return output, [output]
        else:
            return output


@MODELS.register_module()
class PTV3BackboneSimple(nn.Module):
    """Simplified PTV3 backbone with commonly used defaults.
    
    This is a convenience wrapper with sensible defaults for forest/outdoor
    point cloud segmentation. Uses original PTv3 channels for pretrained weight compatibility.
    """
    
    def __init__(
        self,
        in_channels=64,  # Default matches ForestFormer's input_conv output
        out_channels=64,
        grid_size=0.2,  # Default matches ForestFormer's voxel size
        enable_flash=True,
        drop_path=0.3,
        return_blocks=False,
    ):
        super().__init__()
        
        self.backbone = PTV3Backbone(
            in_channels=in_channels,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            # Original PTv3 channels - matches pretrained weights
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(out_channels, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=drop_path,
            shuffle_orders=True,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=enable_flash,
            upcast_attention=False,
            upcast_softmax=False,
            pdnorm_bn=False,
            pdnorm_ln=False,
            pdnorm_decouple=True,
            pdnorm_adaptive=False,
            pdnorm_affine=True,
            pdnorm_conditions=("ForAINet",),
            return_blocks=return_blocks,
            grid_size=grid_size,
        )
        
        self.out_channels = out_channels
        self.return_blocks = return_blocks
    
    def forward(self, x, previous_outputs=None):
        return self.backbone(x, previous_outputs)

