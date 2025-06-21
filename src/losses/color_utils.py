import torch

# RGB to HSV conversion (PyTorch version)
# Input: Tensor of shape (N, 3, H, W) in RGB format, range [0, 1]
# Output: Tensor of shape (N, 3, H, W) in HSV format, H:[0,1], S:[0,1], V:[0,1]
def rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (* , 3, H, W). Got {image.shape}")

    _R: torch.Tensor = image[..., 0, :, :]
    _G: torch.Tensor = image[..., 1, :, :]
    _B: torch.Tensor = image[..., 2, :, :]

    maxc: torch.Tensor = image.max(dim=-3)[0]
    minc: torch.Tensor = image.min(dim=-3)[0]

    H: torch.Tensor = torch.zeros_like(maxc)
    
    delta: torch.Tensor = maxc - minc
    
    # Check if delta is zero to avoid division by zero later
    mask_maxc_eq_minc = (delta == 0)

    # Hue calculation
    mask_r = (_R == maxc) & ~mask_maxc_eq_minc
    H[mask_r] = (_G[mask_r] - _B[mask_r]) / delta[mask_r]

    mask_g = (_G == maxc) & ~mask_maxc_eq_minc
    H[mask_g] = 2.0 + (_B[mask_g] - _R[mask_g]) / delta[mask_g]

    mask_b = (_B == maxc) & ~mask_maxc_eq_minc
    H[mask_b] = 4.0 + (_R[mask_b] - _G[mask_b]) / delta[mask_b]

    H = H / 6.0  # Normalize H to [0, 1]
    H[H < 0] += 1.0 # Handle negative Hue values

    # Saturation calculation
    S: torch.Tensor = torch.zeros_like(maxc)
    mask_maxc_nz = (maxc != 0)
    S[mask_maxc_nz] = delta[mask_maxc_nz] / maxc[mask_maxc_nz]

    # Value calculation (V is just maxc)
    V: torch.Tensor = maxc

    return torch.stack([H, S, V], dim=-3)