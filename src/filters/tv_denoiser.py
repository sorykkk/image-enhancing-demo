import numpy as np
from PIL import Image


def _divergence(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Compute divergence of (px, py) with backward differences."""
    dpx = np.zeros_like(px)
    dpx[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
    dpx[:, 0] = px[:, 0]
    dpx[:, -1] = -px[:, -2]

    dpy = np.zeros_like(py)
    dpy[1:-1, :] = py[1:-1, :] - py[:-2, :]
    dpy[0, :] = py[0, :]
    dpy[-1, :] = -py[-2, :]

    return dpx + dpy


def _gradient(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward-difference gradient."""
    gx = np.zeros_like(u)
    gy = np.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy


def tv_denoise_rof(image_path: str, lam: float = 0.15,
                   n_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Apply Total Variation denoising via the Rudin-Osher-Fatemi model.

    Uses the Chambolle dual projection algorithm to solve:
        min_u  (1/2) * ||u - f||^2  +  lambda * TV(u)

    Operates on YCbCr luminance, preserving chrominance.
    Returns (original_rgb, denoised_rgb) as uint8 numpy arrays.
    """
    img = Image.open(image_path).convert("RGB")
    original = np.array(img)

    ycbcr = np.array(img.convert("YCbCr"), dtype=np.float64)
    f = ycbcr[:, :, 0] / 255.0  # normalise to [0, 1]

    # Chambolle's projection algorithm
    tau = 0.248  # step size (must be <= 1/4 for convergence)
    px = np.zeros_like(f)
    py = np.zeros_like(f)

    for _ in range(n_iter):
        div_p = _divergence(px, py)
        u = f + lam * div_p  # primal update (not clamped yet)

        gx, gy = _gradient(u)
        norm = np.sqrt(gx ** 2 + gy ** 2)
        denom = 1.0 + tau / lam * norm

        px = (px + tau / lam * gx) / denom
        py = (py + tau / lam * gy) / denom

    # Final denoised image
    u = np.clip(f + lam * _divergence(px, py), 0, 1)
    ycbcr[:, :, 0] = u * 255.0

    denoised_img = Image.fromarray(ycbcr.astype(np.uint8), mode="YCbCr").convert("RGB")
    denoised = np.array(denoised_img)

    return original, denoised
