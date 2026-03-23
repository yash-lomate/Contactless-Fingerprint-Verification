"""
Bézier Surface Modeling for Contactless Fingerprint Verification
================================================================
Implements the CL2CB paper pipeline steps 2-3:
  Step 2 – Bézier Surface Modeling (3D parametric surface from control points)
  Step 3 – 3D-to-2D Projection (modeling → view → projection → viewport)

References:
  - Bernstein basis polynomials: B_{i,n}(t) = C(n,i) * t^i * (1-t)^(n-i)
  - Bézier surface: S(u,v) = ΣΣ B_{i,m}(u) * B_{j,n}(v) * P_{i,j}
"""

import numpy as np
import cv2
from scipy.special import comb
from PIL import Image



# 1.  Bernstein Basis Polynomials


def bernstein_poly(i: int, n: int, t: np.ndarray) -> np.ndarray:
    """
    Compute the i-th Bernstein basis polynomial of degree n at parameter t.
    B_{i,n}(t) = C(n, i) * t^i * (1 - t)^(n - i)
    """
    return comb(n, i, exact=True) * (t ** i) * ((1.0 - t) ** (n - i))



# 2.  Bézier Surface Evaluation


def bezier_surface(control_points: np.ndarray,
                   num_u: int = 50,
                   num_v: int = 50) -> np.ndarray:
    """
    Evaluate a Bézier surface patch.

    Parameters
    ----------
    control_points : ndarray of shape (m+1, n+1, 3)
        Grid of 3D control points  P_{i,j} = (x, y, z).
    num_u, num_v : int
        Number of evaluation samples along u and v directions.

    Returns
    -------
    surface : ndarray of shape (num_u, num_v, 3)
        Evaluated (x, y, z) surface coordinates.
    """
    m = control_points.shape[0] - 1  # degree in u
    n = control_points.shape[1] - 1  # degree in v

    u = np.linspace(0.0, 1.0, num_u)
    v = np.linspace(0.0, 1.0, num_v)

    # Pre-compute Bernstein coefficients  ─────────────
    Bu = np.array([bernstein_poly(i, m, u) for i in range(m + 1)])  # (m+1, num_u)
    Bv = np.array([bernstein_poly(j, n, v) for j in range(n + 1)])  # (n+1, num_v)

    # S(u,v) = Σ_i Σ_j  B_{i,m}(u) * B_{j,n}(v) * P_{i,j}
    # Efficient tensor computation:
    #   Bu.T  → (num_u, m+1)
    #   Bv    → (n+1 , num_v)
    #   control_points → (m+1, n+1, 3)
    # result → (num_u, num_v, 3)
    surface = np.einsum('ui,ij k,jv->uvk', Bu.T, control_points, Bv)

    return surface



# 3.  Fit Bézier Surface from Fingerprint Image


def _depth_map_from_image(gray: np.ndarray) -> np.ndarray:
    """
    Approximate a depth-like surface from a grayscale fingerprint image
    using Sobel gradient magnitudes (ridge topology proxy).
    """
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize to [0, 1]  – peaks represent ridge locations
    magnitude = magnitude / (magnitude.max() + 1e-8)
    return magnitude


def fit_bezier_from_image(gray_img: np.ndarray,
                          grid_m: int = 6,
                          grid_n: int = 6,
                          eval_res: int = 100) -> dict:
    """
    Full Bézier surface fitting pipeline for a fingerprint image.

    Steps:
      1. Build a depth map from the grayscale image (gradient magnitude).
      2. Sample an (m×n) grid of control points from the depth map.
      3. Evaluate the Bézier surface.

    Parameters
    ----------
    gray_img  : 2-D uint8 ndarray (grayscale fingerprint)
    grid_m, grid_n : control-point grid dimensions
    eval_res  : number of evaluation points per axis

    Returns
    -------
    dict with keys:
      'control_points' : (grid_m, grid_n, 3)
      'surface'        : (eval_res, eval_res, 3)
      'depth_map'      : (H, W) normalized depth map
    """
    h, w = gray_img.shape[:2]
    depth = _depth_map_from_image(gray_img)

    # Sample control points on a regular grid
    rows = np.linspace(0, h - 1, grid_m, dtype=int)
    cols = np.linspace(0, w - 1, grid_n, dtype=int)

    control_points = np.zeros((grid_m, grid_n, 3), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            control_points[i, j] = [c, r, depth[r, c]]

    # Evaluate surface
    surface = bezier_surface(control_points, num_u=eval_res, num_v=eval_res)

    return {
        'control_points': control_points,
        'surface': surface,
        'depth_map': depth,
    }



# 4.  3D-to-2D Perspective Projection Pipeline


def _build_projection_matrices(image_w: int, image_h: int,
                                fov_deg: float = 60.0,
                                z_near: float = 0.1,
                                z_far: float = 1000.0):
    """
    Construct the classic 3-stage projection matrices:
      Model-View  →  Perspective Projection  →  Viewport Mapping
    """
    # --- Model-View (identity – camera at origin looking down -Z) ---
    model_view = np.eye(4, dtype=np.float64)

    # --- Perspective Projection (symmetric frustum) ---
    aspect = image_w / image_h
    fov_rad = np.deg2rad(fov_deg)
    f = 1.0 / np.tan(fov_rad / 2.0)

    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (z_far + z_near) / (z_near - z_far)
    proj[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    proj[3, 2] = -1.0

    # --- Viewport Mapping ---
    viewport = np.array([
        [image_w / 2, 0, 0, image_w / 2],
        [0, image_h / 2, 0, image_h / 2],
        [0, 0, 0.5, 0.5],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    return model_view, proj, viewport


def project_3d_to_2d(surface: np.ndarray,
                     image_size: tuple = (256, 256)) -> np.ndarray:
    """
    Project a 3D Bézier surface to a 2D image via the full rendering pipeline:
      modeling transformation → view transformation →
      perspective projection → viewport mapping.

    Parameters
    ----------
    surface    : (U, V, 3) array of (x, y, z) surface points
    image_size : (width, height) of the output 2D image

    Returns
    -------
    projected_2d : (U*V, 2) array of 2D pixel coordinates
    output_image : (height, width) uint8 projected fingerprint image
    """
    w, h = image_size
    mv, proj, vp = _build_projection_matrices(w, h)

    U, V, _ = surface.shape
    pts = surface.reshape(-1, 3)

    # Center the surface and push back along Z for the camera
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    z_offset = max(pts_centered[:, :2].max(), abs(pts_centered[:, :2].min())) * 2.5
    pts_centered[:, 2] -= z_offset  # push into -Z

    # Homogeneous coordinates (N, 4)
    ones = np.ones((pts_centered.shape[0], 1))
    pts_h = np.hstack([pts_centered, ones])

    # Pipeline:  Viewport × Projection × ModelView × point
    M = vp @ proj @ mv
    projected = (M @ pts_h.T).T  # (N, 4)

    # Perspective divide
    w_clip = projected[:, 3:4]
    w_clip[w_clip == 0] = 1e-8
    ndc = projected[:, :3] / w_clip

    coords_2d = ndc[:, :2].astype(np.float64)

    # Render into an image
    output = np.zeros((h, w), dtype=np.uint8)
    z_vals = surface.reshape(-1, 3)[:, 2]
    z_norm = ((z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-8) * 255).astype(np.uint8)

    for idx, (px, py) in enumerate(coords_2d):
        xi, yi = int(round(px)), int(round(py))
        if 0 <= xi < w and 0 <= yi < h:
            output[yi, xi] = max(output[yi, xi], z_norm[idx])

    # Slight dilation to fill gaps
    kernel = np.ones((3, 3), np.uint8)
    output = cv2.dilate(output, kernel, iterations=1)

    return coords_2d, output



# 5.  Convenience: end-to-end from image path


def process_fingerprint(image_path: str,
                        grid_m: int = 6,
                        grid_n: int = 6,
                        eval_res: int = 100,
                        output_size: tuple = (256, 256)) -> dict:
    """
    Run the full Bézier pipeline on a fingerprint image file.

    Returns dict with:
      'gray'            : original grayscale image
      'depth_map'       : gradient-based depth approximation
      'control_points'  : sampled control-point grid
      'surface'         : evaluated 3D Bézier surface
      'projected_coords': 2D projected coordinates
      'projected_image' : 2D output image after perspective projection
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = cv2.resize(img, output_size)
    result = fit_bezier_from_image(img, grid_m, grid_n, eval_res)

    coords, proj_img = project_3d_to_2d(result['surface'], output_size)

    result['gray'] = img
    result['projected_coords'] = coords
    result['projected_image'] = proj_img
    return result
