from typing import List

import torch
from torch import Tensor
import torch.nn.functional as F
 

def get_matched_ref_features(features: List[Tensor], ref_features: List[Tensor]) -> List[Tensor]:
    """
    Get matched reference features for one class.
    """
    matched_ref_features = []
    for layer_id in range(len(features)):
        feature = features[layer_id]
        B, C, H, W = feature.shape
        feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
        feature_n = F.normalize(feature, p=2, dim=1)
        coreset = ref_features[layer_id]  # (N2, C)
        coreset_n = F.normalize(coreset, p=2, dim=1)
        dist = feature_n @ coreset_n.T
        cidx = torch.argmax(dist, dim=1)
        index_feats = coreset[cidx]
        index_feats = index_feats.reshape(B, H, W, C).permute(0, 3, 1, 2)
        matched_ref_features.append(index_feats)
    
    return matched_ref_features


def _get_spatial_distance(pos_ids, matched_ids, size=(16, 16)):
    H, W = size[0], size[1]
    num_tokens_one_image = H * W
    ref_image_ids = matched_ids // num_tokens_one_image
    pos_ids_in_image = matched_ids % num_tokens_one_image
    h_ids_in_image = pos_ids_in_image // W
    w_ids_in_image = pos_ids_in_image % W
    hw_ids_in_image = torch.stack([h_ids_in_image, w_ids_in_image], dim=-1)
    hw_ids_in_image = hw_ids_in_image.to(torch.float)
    spatial_distance = torch.abs(pos_ids.reshape(-1, 2) - hw_ids_in_image)
    l1_distance = spatial_distance[:, 0] + spatial_distance[:, 1]
    l1_distance = l1_distance.to(torch.long)
    
    return l1_distance


def get_matched_ref_features_and_ids(features: List[Tensor], ref_features: List[Tensor]) -> List[Tensor]:
    """
    Get matched reference features for one class.
    """
    feature = features[0]
    _, _, H, W = feature.shape
    x = torch.arange(W, dtype=torch.float)  
    y = torch.arange(H, dtype=torch.float)  
    xx, yy = torch.meshgrid(x, y)
    xy = torch.stack([xx, yy], dim=-1)  # (h, w, 2), (pos_in_h, pos_in_w)
    xy = xy.to(device=feature.device)
    
    matched_ref_features = []
    matched_spatial_distances = []
    for layer_id in range(len(features)):
        feature = features[layer_id]
        B, C, H, W = feature.shape
        feature = feature.permute(0, 2, 3, 1).reshape(-1, C).contiguous()  # (N1, C)
        feature_n = F.normalize(feature, p=2, dim=1)
        coreset = ref_features[layer_id]  # (N2, C)
        coreset_n = F.normalize(coreset, p=2, dim=1)
        
        dist = feature_n @ coreset_n.T
        cidx = torch.argmax(dist, dim=1)
        index_feats = coreset[cidx]
        index_feats = index_feats.reshape(B, H, W, C).permute(0, 3, 1, 2)
        matched_ref_features.append(index_feats)
        spatial_distance = _get_spatial_distance(xy, cidx, (H, W))   
        matched_spatial_distances.append(spatial_distance.unsqueeze(0))
    
    return matched_ref_features, matched_spatial_distances


def get_residual_features(features: List[Tensor], ref_features: List[Tensor], pos_flag: bool = False) -> List[Tensor]:
    residual_features = []
    for layer_id in range(len(features)):
        fi = features[layer_id]  # (B, dim, h, w)
        pi = ref_features[layer_id]  # (B, dim, h, w)
        
        if not pos_flag:
            ri = fi - pi
        else:
            ri = F.mse_loss(fi, pi, reduction='none')
        residual_features.append(ri)
    
    return residual_features