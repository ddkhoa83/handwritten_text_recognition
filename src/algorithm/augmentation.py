"""NumPy image translation and rotation with manual bilinear sampling."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def augment_translate(X: FloatArray, ty: float, tx: float,
                      fill: float = 0.0) -> FloatArray:
    """Translate every image by the same (ty, tx), preserving input shape.

    Accept (N,784), (H,W), (N,H,W), or (N,C,H,W). Return float64.
    Positive ty moves content down; positive tx moves content right.
    Integer shifts use slicing; fractional shifts use bilinear sampling.
    """
    X_arr, original_ndim = _prepare_images(X)
    ty, tx, fill = float(ty), float(tx), float(fill)
    if not np.isfinite([ty, tx, fill]).all():
        raise ValueError("ty, tx, and fill must be finite.")
    h, w = X_arr.shape[-2:]

    # No part of the source image overlaps the output frame.
    if abs(ty) >= h or abs(tx) >= w:
        return _restore_shape(np.full_like(X_arr, fill), original_ndim)

    if ty.is_integer() and tx.is_integer():
        iy, ix = int(ty), int(tx)
        out = np.full_like(X_arr, fill)
        if iy >= 0:
            src_y0, src_y1 = 0, h - iy
            dst_y0, dst_y1 = iy, h
        else:
            src_y0, src_y1 = -iy, h
            dst_y0, dst_y1 = 0, h + iy
        if ix >= 0:
            src_x0, src_x1 = 0, w - ix
            dst_x0, dst_x1 = ix, w
        else:
            src_x0, src_x1 = -ix, w
            dst_x0, dst_x1 = 0, w + ix
        out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = (
            X_arr[:, :, src_y0:src_y1, src_x0:src_x1]
        )
        return _restore_shape(out, original_ndim)

    grid_y, grid_x = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64), indexing="ij"
    )
    out = _bilinear_sample(X_arr, grid_y - ty, grid_x - tx, fill)
    return _restore_shape(out, original_ndim)


def augment_rotate(X: FloatArray, max_angle: float = 10.0,
                   probability: float = 0.5,
                   rng: np.random.Generator | None = None,
                   fill: float = 0.0) -> FloatArray:
    """Randomly rotate each selected sample about the image center.

    Each sample is selected independently with the given probability.
    Selected angles are uniform in [-max_angle, max_angle] degrees.
    Positive angles rotate content counterclockwise. Output retains the
    input shape and uses float64. Unselected samples retain converted values.
    Pass a seeded generator for reproducible sequences of augmentations.
    """
    max_angle, probability, fill = float(max_angle), float(probability), float(fill)
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be finite and in [0, 1].")
    if not np.isfinite(max_angle) or max_angle < 0:
        raise ValueError("max_angle must be finite and nonnegative.")
    if not np.isfinite(fill):
        raise ValueError("fill must be finite.")
    if rng is None:
        rng = np.random.default_rng()

    X_arr, original_ndim = _prepare_images(X)
    n = len(X_arr)
    out = X_arr.copy()
    if n == 0 or max_angle == 0 or probability == 0:
        return _restore_shape(out, original_ndim)
    gate = rng.random(n) < probability
    if gate.any():
        subset = X_arr[gate]
        angles = rng.uniform(-max_angle, max_angle, size=len(subset))
        out[gate] = _rotate_angles(subset, angles, fill)
    return _restore_shape(out, original_ndim)


def translate_random(X: FloatArray, max_ty: int, max_tx: int,
                     rng: np.random.Generator,
                     fill: float = 0.0) -> FloatArray:
    """Apply independent integer shifts to each sample, preserving shape.

    dy is uniform on {-max_ty, ..., max_ty}; dx is analogous.
    All channels of a sample use the same shift.
    """
    for name, value in (("max_ty", max_ty), ("max_tx", max_tx)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer.")
    fill = float(fill)
    if not np.isfinite(fill):
        raise ValueError("fill must be finite.")
    X_arr, original_ndim = _prepare_images(X)
    n = len(X_arr)
    tys = rng.integers(-max_ty, max_ty + 1, size=n)
    txs = rng.integers(-max_tx, max_tx + 1, size=n)
    out = np.empty_like(X_arr)
    for i in range(n):
        out[i:i + 1] = augment_translate(
            X_arr[i:i + 1], float(tys[i]), float(txs[i]), fill=fill
        )
    return _restore_shape(out, original_ndim)


def _rotate_angles(X_arr: FloatArray, angles: FloatArray,
                   fill: float) -> FloatArray:
    """Sample a batch using one rotation angle in degrees per sample."""
    n, c, h, w = X_arr.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    grid_y, grid_x = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64), indexing="ij"
    )
    theta = np.deg2rad(angles)
    cos_t = np.cos(theta)[:, None, None, None]
    sin_t = np.sin(theta)[:, None, None, None]
    dy = grid_y[None, None] - cy
    dx = grid_x[None, None] - cx
    src_y = cos_t * dy + sin_t * dx + cy
    src_x = -sin_t * dy + cos_t * dx + cx
    return _bilinear_sample(X_arr, src_y, src_x, fill)


def _bilinear_sample(X_arr: FloatArray, src_y: FloatArray,
                     src_x: FloatArray, fill: float) -> FloatArray:
    """Sample coordinates shaped (H,W) or (N,1,H,W), with constant padding."""
    n, c, h, w = X_arr.shape
    y0 = np.floor(src_y).astype(np.int64)
    x0 = np.floor(src_x).astype(np.int64)
    wy, wx = src_y - y0, src_x - x0
    y0c = np.clip(y0, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)

    valid_y0 = (y0 >= 0) & (y0 < h)
    valid_y1 = (y0 + 1 >= 0) & (y0 + 1 < h)
    valid_x0 = (x0 >= 0) & (x0 < w)
    valid_x1 = (x0 + 1 >= 0) & (x0 + 1 < w)
    w00 = (1 - wy) * (1 - wx) * valid_y0 * valid_x0
    w01 = (1 - wy) * wx * valid_y0 * valid_x1
    w10 = wy * (1 - wx) * valid_y1 * valid_x0
    w11 = wy * wx * valid_y1 * valid_x1

    # Pair each sample's coordinates with that sample, across all channels.
    batch_idx = np.arange(n)[:, None, None, None]
    channel_idx = np.arange(c)[None, :, None, None]
    V00 = X_arr[batch_idx, channel_idx, y0c, x0c]
    V01 = X_arr[batch_idx, channel_idx, y0c, x1c]
    V10 = X_arr[batch_idx, channel_idx, y1c, x0c]
    V11 = X_arr[batch_idx, channel_idx, y1c, x1c]
    total = w00 + w01 + w10 + w11
    out = V00 * w00 + V01 * w01 + V10 * w10 + V11 * w11
    return np.where(total > 1e-12, out + fill * (1 - total), fill)


def _prepare_images(X: FloatArray) -> tuple[FloatArray, int]:
    """Normalize images to (N,C,H,W).

    Supported layouts:
        (N,784): flattened MNIST images.
        (H,W): a single grayscale image.
        (N,H,W): a batch of grayscale images.
        (N,C,H,W): a batch with explicit channels.

    Returns
    -------
    tuple[FloatArray, int]
        Normalized images and a layout marker.
        Marker -1 identifies flattened MNIST input.
        Other markers represent the original number of dimensions.
    """
    arr = np.asarray(X, dtype=np.float64)
    original_ndim = arr.ndim

    if original_ndim == 2:
        if arr.shape[1] == 28 * 28:
            # Preserve the batch size, including an empty batch.
            arr = arr.reshape(arr.shape[0], 1, 28, 28)
            original_ndim = -1
        else:
            arr = arr[None, None, :, :]

    elif original_ndim == 3:
        arr = arr[:, None, :, :]

    elif original_ndim != 4:
        raise ValueError(
            "X must have shape (N,784), (H,W), "
            f"(N,H,W), or (N,C,H,W); got {arr.shape}."
        )

    if any(size == 0 for size in arr.shape[1:]):
        raise ValueError("Channels, height, and width must be positive.")

    return arr, original_ndim


def _restore_shape(arr: FloatArray, original_ndim: int) -> FloatArray:
    """Restore the input layout using the marker from _prepare_images."""
    if original_ndim == -1:
        return arr.reshape(arr.shape[0], 28 * 28)

    if original_ndim == 2:
        return arr[0, 0]

    if original_ndim == 3:
        return arr[:, 0]

    return arr

def augment_random(
    X: FloatArray,
    max_ty: int = 1,
    max_tx: int = 1,
    max_angle: float = 5.0,
    rotation_probability: float = 0.5,
    rng: np.random.Generator | int | None = None,
    fill: float = 0.0,
    *,
    probability: float = 0.3,
) -> FloatArray:
    """Apply at most one randomly selected transform to each selected image.

    Parameters
    ----------
    X : FloatArray
        Pixel arrays shaped (N,784), (H,W), (N,H,W), or (N,C,H,W).
        Apply augmentation before any per-feature standardization.
    max_ty, max_tx : int, default=1
        Nonnegative integer bounds for vertical and horizontal shifts.
    max_angle : float, default=5.0
        Nonnegative rotation bound in degrees.
    rotation_probability : float, default=0.5
        Conditional probability of rotation given that an image is selected.
        Otherwise that image receives a random integer translation.
    rng : numpy.random.Generator | int | None
        Generator or seed. Reuse a Generator across training batches.
        Passing the same integer repeatedly restarts the random sequence.
    fill : float, default=0.0
        Finite background value in the same scale as X.
    probability : float, keyword-only, default=0.3
        Independent probability of selecting each image for augmentation.
        A selected translation can still be (0,0). Zero disables augmentation.

    Returns
    -------
    FloatArray
        A new float64 array with the input layout. Unselected images retain
        their converted values. Labels and sample order are preserved.

    Notes
    -----
    These mild defaults are a starting point for validation experiments.
    Selection is per image, replacing the former whole-batch method choice.
    Only apply this function to training inputs.
    """
    probability = float(probability)
    rotation_probability = float(rotation_probability)
    for name, value in (
        ("probability", probability),
        ("rotation_probability", rotation_probability),
    ):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1].")
    for name, value in (("max_ty", max_ty), ("max_tx", max_tx)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer.")
    max_angle, fill = float(max_angle), float(fill)
    if not np.isfinite(max_angle) or max_angle < 0:
        raise ValueError("max_angle must be finite and nonnegative.")
    if not np.isfinite(fill):
        raise ValueError("fill must be finite.")

    rng = np.random.default_rng(rng)
    images, layout = _prepare_images(X)
    out = images.copy()
    if probability == 0 or len(images) == 0:
        return _restore_shape(out, layout)

    selected = rng.random(len(images)) < probability
    rotate = selected & (rng.random(len(images)) < rotation_probability)
    translate = selected & ~rotate

    if rotate.any() and max_angle > 0:
        angles = rng.uniform(-max_angle, max_angle, size=int(rotate.sum()))
        out[rotate] = _rotate_angles(images[rotate], angles, fill)
    if translate.any() and (max_ty > 0 or max_tx > 0):
        out[translate] = translate_random(
            images[translate], max_ty, max_tx, rng=rng, fill=fill
        )
    return _restore_shape(out, layout)