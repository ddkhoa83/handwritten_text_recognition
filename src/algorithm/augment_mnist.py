import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

def translate_images(X: FloatArray,
                    ty: float, tx: float,
                    fill: float = 0.0) -> FloatArray:
    """
    Translate a batch of images by (ty, tx) pixels

    Positive by shifts contents DOWN, positive tx shifts content RIGHT
    Subpixel shifts use bilinear interpolation; integer shifts use slicing

    Parameters
    -----------
    X: ndarray, shape (N, C, H, W) or (N, H, W) or (H, W)
        Input images. Any numeric dtype; internally cast to float64.
    ty: float
        Vertical shift in pixels. Positive = content moves down
        Negative = content moves up
    tx: float
        Horizontal shift in pixels. Positive = content moves right.
    fill: float, default = 0.0
        Value used for pixels that shift in from outside the image.

    Returns
    --------
    ndarray
        Translated images, same shape as input

    Examples
    -------
    >>> X = np.random.rand(4, 1, 28, 28)
    >>> Y = translate_images(X, ty = 2, tx = -1) #integer shift
    >>> Y = translate_images(X, ty = 0.5, tx = 0.0) #subpixel shift
    >>> Y = translate_images(X[0, 0], 3, 3) #single 2D image
    """
    X_arr = np.asarray(X, dtype = np.float64)
    original_shape = X.shape
    original_ndim = X_arr.ndim

    #Normalize to (N, C, H, W)
    if original_ndim == 2:
        X_arr = X_arr[None, None, :, :]
    elif original_ndim == 3:
        X_arr = X_arr[:, None, :, :]
    elif original_ndim != 4:
        raise ValueError(
            f"X must be 2D, 3D, 4D; got shape {original_shape}."
        )

    ty = float(ty)
    tx = float(tx)
    n, c, h, w = X_arr.shape

    #Fast path: integer shift within bounds -> pure slicing
    if ty.is_integer() and tx.is_integer():
        ty, ix = int(ty), int(tx)
        if abs(iy) < h and abs(ix) < w:
            out = np.full_like(X_arr, fill)
            #Source & destination slices
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
            out[:, :, dst_y0: dst_y1: dst_x1] = X_arr[:, :, src_y0:src_y1: src_x0: src_x1]
            return _restore_shape(out, original_ndim)

    
                
       
            