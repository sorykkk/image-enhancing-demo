import numpy as np
from PIL import Image


def equalize_histogram(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply histogram equalization to a color image.

    Converts to YCbCr, equalizes the Y (luminance) channel, converts back to RGB.
    Returns (original_rgb, equalized_rgb) as uint8 numpy arrays.
    """
    img = Image.open(image_path).convert("RGB")
    original = np.array(img)

    ycbcr = np.array(img.convert("YCbCr"), dtype=np.float64)
    y_channel = ycbcr[:, :, 0]

    # Compute histogram and CDF
    hist, _ = np.histogram(y_channel.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    total_pixels = y_channel.size

    # Apply equalization: s_k = round((cdf(k) - cdf_min) / (M*N - cdf_min) * 255)
    lut = np.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255).astype(np.uint8)
    ycbcr[:, :, 0] = lut[y_channel.astype(np.uint8)]

    equalized_img = Image.fromarray(ycbcr.astype(np.uint8), mode="YCbCr").convert("RGB")
    equalized = np.array(equalized_img)

    return original, equalized