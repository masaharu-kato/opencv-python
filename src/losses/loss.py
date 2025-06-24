from typing import Literal
from jaxtyping import Float as F
import lpips
import piqa
from torch import Tensor, nn, cuda

DEVICE = "cuda" if cuda.is_available() else "cpu"

class MaskedL1:
    def __init__(self):
        self.l1_loss = nn.L1Loss(reduction='none')

    def __call__(self, pred: F[Tensor, "B C H W"], target: F[Tensor, "B C H W"], mask: F[Tensor, "B 1 H W"]) -> F[Tensor, ""]:
        """
        Compute the masked L1 loss between the predicted and target tensors.

        Args:
            pred (BCHW): The predicted tensor.
            target (BCHW): The target tensor.
            mask (B1HW): The gray mask tensor, where 1 indicates valid pixels and 0 indicates invalid pixels.

        Returns:
            torch.Tensor: The computed masked L1 loss.
        """
        loss: F[Tensor, "B C H W"] = self.l1_loss(pred, target)
        loss_gray: F[Tensor, "B 1 H W"] = loss.mean(dim=1, keepdim=True)  # Average over color channels
        return ((loss_gray * mask).sum() / (mask.sum() + 1e-8))


LPIPS_MODEL = Literal["alex", "vgg", "squeeze"]

class LPIPS:
    def __init__(self, lpips_model: LPIPS_MODEL):
        self.lipis_loss = lpips.LPIPS(net=lpips_model, spatial=False).to(DEVICE)
        self.lipis_loss.eval()  # Set to evaluation mode

    def __call__(self, pred: F[Tensor, "B C H W"], target: F[Tensor, "B C H W"]) -> F[Tensor, ""]:
        """
        Compute the LPIPS loss between the predicted and target tensors.

        Args:
            pred (BCHW): The predicted tensor.
            target (BCHW): The target tensor.

        Returns:
            torch.Tensor: The computed LPIPS loss.
        """
        scaled_pred = pred * 2.0 - 1.0  # Scale to [-1, 1] for LPIPS
        scaled_target = target * 2.0 - 1.0  # Scale to [-1, 1] for LPIPS
        return self.lipis_loss(scaled_pred, scaled_target).mean()


class SSIM:
    def __init__(self):
        self.ssim_loss = piqa.SSIM(n_channels=3).to(DEVICE)

    def __call__(self, pred: F[Tensor, "B C H W"], target: F[Tensor, "B C H W"]) -> F[Tensor, ""]:
        """
        Compute the SSIM between the predicted and target tensors.

        Args:
            pred (BCHW): The predicted tensor.
            target (BCHW): The target tensor.

        Returns:
            torch.Tensor: The computed SSIM loss.
        """
        return self.ssim_loss(pred, target).mean()
