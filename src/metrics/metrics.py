import torch
import torch.nn.functional as F


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """
    pred, target: expected shape [B,C,H,W,D] or compatible
    pred can be probabilistic; target can be binary/probabilistic
    """
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    if union.item() == 0:
        return 1.0

    dice = (2.0 * intersection + eps) / (union + eps)
    return float(dice.item())


def _extract_surface_coords(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: [B, C, H, W, D] or [H, W, D]
    returns: [N, 3] surface voxel coordinates on CPU
    """
    if mask.dim() == 5:
        mask = mask[0, 0]
    elif mask.dim() == 4:
        mask = mask[0]

    mask = (mask > 0).to(torch.float32).cpu()

    if mask.sum() == 0:
        return torch.empty((0, 3), dtype=torch.float32)

    mask_5d = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W,D]
    eroded = 1.0 - F.max_pool3d(1.0 - mask_5d, kernel_size=3, stride=1, padding=1)
    surface = mask_5d - eroded
    surface = (surface > 0).squeeze(0).squeeze(0)

    coords = torch.nonzero(surface, as_tuple=False).to(torch.float32)
    return coords.cpu()


def _chunked_min_cdist(a: torch.Tensor, b: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """
    a: [N, 3] on CPU
    b: [M, 3] on CPU
    returns: min distance from each point in a to set b -> [N]
    """
    if a.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)
    if b.numel() == 0:
        return torch.full((a.shape[0],), float("inf"), dtype=torch.float32)

    mins = []
    for i in range(0, a.shape[0], chunk_size):
        a_chunk = a[i:i + chunk_size]
        d = torch.cdist(a_chunk, b)
        mins.append(d.min(dim=1).values)
    return torch.cat(mins, dim=0)


def nsd_score(pred: torch.Tensor, target: torch.Tensor, tolerance: float = 1.0, chunk_size: int = 4096) -> float:
    """
    Memory-safe NSD approximation:
    - uses surface voxels only
    - computes distances on CPU
    - uses chunked cdist
    """
    pred = (pred > 0.5).to(torch.float32)
    target = (target > 0.5).to(torch.float32)

    pred_surface = _extract_surface_coords(pred)
    target_surface = _extract_surface_coords(target)

    if pred_surface.shape[0] == 0 and target_surface.shape[0] == 0:
        return 1.0
    if pred_surface.shape[0] == 0 or target_surface.shape[0] == 0:
        return 0.0

    pred_to_target = _chunked_min_cdist(pred_surface, target_surface, chunk_size=chunk_size)
    target_to_pred = _chunked_min_cdist(target_surface, pred_surface, chunk_size=chunk_size)

    pred_hit = (pred_to_target <= tolerance).float().mean()
    target_hit = (target_to_pred <= tolerance).float().mean()

    nsd = 0.5 * (pred_hit + target_hit)
    return float(nsd.item())