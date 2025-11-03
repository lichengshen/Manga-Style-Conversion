import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from noise import pnoise2
from tqdm import tqdm

def apply_rotation(X: np.ndarray, Y: np.ndarray, rad: float):
    """Rotate coordinate grid by angle rad (in radians)."""
    Xr = X * np.cos(rad) - Y * np.sin(rad)
    Yr = X * np.sin(rad) + Y * np.cos(rad)
    return Xr, Yr

def threshold_and_invert(pattern: np.ndarray, threshold: float, invert: bool) -> np.ndarray:
    """Apply thresholding to image in [0,1] and invert if needed. Returns uint8 0/255."""
    binary = pattern > threshold
    if invert:
        binary = ~binary
    return (binary.astype(np.uint8) * 255)

def dot_tone(freq: float, threshold: float, rad: float, shape, invert: bool = False) -> np.ndarray:
    """
    Generate a dot pattern with specified frequency and rotation.
    """
    h, w = shape
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    X, Y = np.meshgrid(x, y)
    Xr, Yr = apply_rotation(X, Y, rad)

    pattern = np.sin(freq * np.pi * Xr) * np.sin(freq * np.pi * Yr)
    pattern = (pattern + 1) / 2  # normalize to [0, 1]
    return threshold_and_invert(pattern, threshold, invert)

def line_tone(freq: float, threshold: float, rad: float, shape, invert: bool = False) -> np.ndarray:
    """
    Generate a line pattern with specified frequency and rotation.
    """
    h, w = shape
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    X, Y = np.meshgrid(x, y)
    Xr, Yr = apply_rotation(X, Y, rad)

    pattern = np.sin(freq * np.pi * Xr)
    pattern = (pattern + 1) / 2  # normalize to [0, 1]
    return threshold_and_invert(pattern, threshold, invert)

def grid_tone(freq: float, threshold: float, rad: float, shape, invert: bool = False) -> np.ndarray:
    """
    Generate a grid pattern (crosshatch) with specified frequency and rotation.
    """
    h, w = shape
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    X, Y = np.meshgrid(x, y)
    Xr, Yr = apply_rotation(X, Y, rad)

    pattern_x = np.sin(freq * np.pi * Xr)
    pattern_x = (pattern_x + 1) / 2
    pattern_y = np.sin(freq * np.pi * Yr)
    pattern_y = (pattern_y + 1) / 2

    # combine patterns into a grid
    pattern = np.minimum(pattern_x > threshold, pattern_y > threshold)
    # pattern is boolean; threshold_and_invert expects [0,1]-like input, but we can pass boolean directly
    return threshold_and_invert(pattern.astype(float), threshold, invert)

def sand_tone(threshold: float, shape, invert: bool = False,
              noise_type: str = "white", **kwargs) -> np.ndarray:
    """
    Generate a sand/noise tone.

    noise_type:
        - 'white'
        - 'gaussian' (mean, std)
        - 'perlin' (scale)
    """
    h, w = shape

    if noise_type == "white":
        noise = np.random.rand(h, w)

    elif noise_type == "gaussian":
        mean = kwargs.get("mean", 0.5)
        std = kwargs.get("std", 0.15)
        noise = np.random.normal(mean, std, (h, w))
        noise = np.clip(noise, 0, 1)

    elif noise_type == "perlin":
        scale = kwargs.get("scale", 50)
        noise = np.zeros((h, w))
        for yy in range(h):
            for xx in range(w):
                noise[yy, xx] = pnoise2(xx / scale, yy / scale, octaves=1)
        # normalize to [0,1]
        noise = (noise - noise.min()) / (noise.max() - noise.min())

    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")

    return threshold_and_invert(noise, threshold, invert)

def directional_noise_tone(freq: float,
                           threshold: float,
                           rad: float,
                           scale: float,
                           shape,
                           invert: bool = False) -> np.ndarray:
    """
    Generate a directional noise pattern with specified frequency and rotation.
    """
    h, w = shape
    img = np.zeros((h, w), dtype=np.float32)

    # generate per-pixel
    for yy in range(h):
        for xx in range(w):
            noise_val = pnoise2(xx / scale, yy / scale, octaves=1)
            # directional wave
            angle_wave = np.sin(
                freq * (xx * np.cos(rad) + yy * np.sin(rad)) * 2 * np.pi / max(h, w)
            )
            img[yy, xx] = noise_val + angle_wave

    # normalize to [0,1]
    img = (img - img.min()) / (img.max() - img.min())
    return threshold_and_invert(img, threshold, invert)

def preview_examples():
    """Show a simple 2x4 grid of example tones."""
    shape = (512, 512)
    examples = {
        "dot_tone": dot_tone(freq=10, threshold=0.2, rad=0, shape=shape),
        "line_tone": line_tone(freq=10, threshold=0.2, rad=np.pi / 4, shape=shape),
        "grid_tone": grid_tone(freq=10, threshold=0.1, rad=np.pi / 6, shape=shape),
        "sand_tone_white": sand_tone(threshold=0.2, shape=shape),
        "sand_tone_gaussian": sand_tone(
            threshold=0.2, shape=shape, noise_type="gaussian", mean=0.5, std=0.4
        ),
        "sand_tone_perlin": sand_tone(
            threshold=0.5, shape=shape, noise_type="perlin", scale=5
        ),
        "directional_noise_tone": directional_noise_tone(
            freq=20, threshold=0.35, rad=np.pi / 6, scale=50, shape=shape
        ),
    }

    plt.figure(figsize=(12, 8))
    for i, (name, img) in enumerate(examples.items()):
        plt.subplot(2, 4, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def generate_and_save_all(save_dir: str = "./screentones_gen"):
    """
    Generate a set of tones for multiple thresholds and save them as PNGs.
    """
    os.makedirs(save_dir, exist_ok=True)

    shape = (2000, 2000)
    thresholds = [i / 10 for i in range(11)]  # 0.0 ... 1.0

    for thresh in tqdm(thresholds, desc="Generating screentones"):
        # dots
        tone_img = dot_tone(freq=200, threshold=thresh, rad=0, shape=shape)
        cv2.imwrite(f"{save_dir}/dot_{thresh:.1f}.png", tone_img)

        # lines – multiple angles
        tone_img = line_tone(freq=200, threshold=thresh, rad=0, shape=shape)
        cv2.imwrite(f"{save_dir}/line0_{thresh:.1f}.png", tone_img)

        tone_img = line_tone(freq=200, threshold=thresh, rad=np.pi / 4, shape=shape)
        cv2.imwrite(f"{save_dir}/line45_{thresh:.1f}.png", tone_img)

        tone_img = line_tone(freq=200, threshold=thresh, rad=np.pi / 2, shape=shape)
        cv2.imwrite(f"{save_dir}/line90_{thresh:.1f}.png", tone_img)

        tone_img = line_tone(freq=200, threshold=thresh, rad=3 * np.pi / 4, shape=shape)
        cv2.imwrite(f"{save_dir}/line135_{thresh:.1f}.png", tone_img)

        # grid
        tone_img = grid_tone(freq=200, threshold=thresh, rad=0, shape=shape)
        cv2.imwrite(f"{save_dir}/grid_{thresh:.1f}.png", tone_img)

        # sand (white)
        tone_img = sand_tone(threshold=thresh, shape=shape)
        cv2.imwrite(f"{save_dir}/sand-white_{thresh:.1f}.png", tone_img)

        # sand (perlin)
        tone_img = sand_tone(threshold=thresh, shape=shape, noise_type="perlin", scale=5)
        cv2.imwrite(f"{save_dir}/sand-perlin_{thresh:.1f}.png", tone_img)

        # directional noise
        tone_img = directional_noise_tone(
            freq=150, threshold=thresh, rad=np.pi / 6, scale=10, shape=shape
        )
        cv2.imwrite(f"{save_dir}/directional-noise_{thresh:.1f}.png", tone_img)


def main():
    # show preview
    # preview_examples()

    # save full-res tones
    generate_and_save_all("./screentones_gen")


if __name__ == "__main__":
    main()
