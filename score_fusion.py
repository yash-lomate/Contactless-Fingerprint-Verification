"""
Score-Level Fusion for Contactless Fingerprint Verification
=============================================================
Implements the CL2CB paper pipeline steps 4-6:
  Step 4 – CNN Feature Extraction (Siamese-style embedding + L2 distance)
  Step 5 – Minutiae-Based Feature Extraction (ridge endings & bifurcations)
  Step 6 – Score-Level Fusion (weighted combination of both branches)
"""

import numpy as np
import cv2
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize



# STEP 5 — MINUTIAE EXTRACTION  (Traditional Branch)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu binarisation → thin skeleton suitable for crossing-number."""
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = (binary / 255).astype(np.uint8)
    return binary


def _thin(binary: np.ndarray) -> np.ndarray:
    """Morphological thinning (skeletonisation) of a binary ridge map."""
    skeleton = skeletonize(binary).astype(np.uint8)
    return skeleton


def _crossing_number(skeleton: np.ndarray):
    """
    Crossing-number technique for minutiae detection.

    For every skeleton pixel p, the crossing number CN is:
        CN(p) = 0.5 * Σ |p_{i} - p_{i+1}|   (i = 1..8, circular)

    CN == 1  →  ridge ending
    CN == 3  →  bifurcation

    Returns
    -------
    endings      : list of (row, col) tuples
    bifurcations : list of (row, col) tuples
    """
    h, w = skeleton.shape
    endings = []
    bifurcations = []

    # 8-connected neighbour offsets (clockwise from top-left)
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0,  1),
               (1,  1),  (1, 0), (1, -1),
               (0, -1)]

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if skeleton[r, c] == 0:
                continue

            neighbours = [skeleton[r + dr, c + dc] for dr, dc in offsets]
            cn = 0.0
            for k in range(len(neighbours)):
                cn += abs(int(neighbours[k]) - int(neighbours[(k + 1) % len(neighbours)]))
            cn *= 0.5

            if cn == 1:
                endings.append((r, c))
            elif cn == 3:
                bifurcations.append((r, c))

    return endings, bifurcations


def extract_minutiae(gray_img: np.ndarray) -> dict:
    """
    Full minutiae extraction pipeline.

    Parameters
    ----------
    gray_img : 2-D uint8 ndarray (grayscale fingerprint)

    Returns
    -------
    dict with keys:
      'endings'       : list of (row, col) ridge endings
      'bifurcations'  : list of (row, col) bifurcations
      'all_minutiae'  : combined list
      'skeleton'      : the thinned binary image
    """
    binary = _binarize(gray_img)
    skeleton = _thin(binary)
    endings, bifurcations = _crossing_number(skeleton)

    return {
        'endings': endings,
        'bifurcations': bifurcations,
        'all_minutiae': endings + bifurcations,
        'skeleton': skeleton,
    }


def minutiae_match_score(minutiae1: list, minutiae2: list,
                         max_distance: float = 20.0) -> float:
    """
    Compute a normalised matching score between two minutiae sets
    using nearest-neighbour distance in (row, col) space.

    A pair is considered matched if the Euclidean distance < max_distance.
    Score = matched_pairs / max(len(set1), len(set2))

    Returns
    -------
    score : float in [0, 1]  (1 = perfect match)
    """
    if len(minutiae1) == 0 or len(minutiae2) == 0:
        return 0.0

    pts1 = np.array(minutiae1, dtype=np.float64)
    pts2 = np.array(minutiae2, dtype=np.float64)

    dists = cdist(pts1, pts2, metric='euclidean')   # (M, N)

    matched = 0
    used2 = set()
    for i in range(len(pts1)):
        sorted_j = np.argsort(dists[i])
        for j in sorted_j:
            if j not in used2 and dists[i, j] < max_distance:
                matched += 1
                used2.add(j)
                break

    score = matched / max(len(minutiae1), len(minutiae2))
    return score



# STEP 4 — CNN FEATURE EXTRACTION  (Deep Learning Branch)


def get_cnn_embedding(image_path: str, model_path: str = 'finger_model.h5',
                      image_size: int = 64):
    """
    Extract a 256-dim embedding from the Dense(256) layer of the
    existing Sequential CNN classifier.

    Parameters
    ----------
    image_path : path to fingerprint image
    model_path : path to the trained .h5 model
    image_size : spatial dimension expected by the model

    Returns
    -------
    embedding : 1-D ndarray of length 256
    """
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from tensorflow.keras.models import load_model, Model

    full_model = load_model(model_path, compile=False)

    # Build a feature-extractor that outputs the Dense(256) layer
    # The Dense(256) is the second-to-last layer (before the softmax)
    feature_layer = full_model.layers[-3]  # Dense(256, relu)
    extractor = Model(inputs=full_model.input, outputs=feature_layer.output)

    from PIL import Image as PILImage
    img = PILImage.open(image_path).resize((image_size, image_size))
    arr = np.array(img).reshape(1, image_size, image_size, 3).astype('float32') / 255.0

    embedding = extractor.predict(arr, verbose=0).flatten()
    return embedding


def cnn_similarity_score(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute a similarity score based on L2 (Euclidean) distance
    between two CNN embeddings.

    Score is normalised to [0, 1] using an exponential decay:
        score = exp(-distance / scale)

    Returns
    -------
    score : float in [0, 1]  (1 = identical embeddings)
    """
    distance = np.linalg.norm(emb1 - emb2)
    # Scale factor chosen empirically so that distance ~5 → score ~0.5
    scale = 7.0
    score = np.exp(-distance / scale)
    return float(score)



# STEP 6 — SCORE-LEVEL FUSION


def fuse_scores(cnn_score: float,
                minutiae_score: float,
                alpha: float = 0.5) -> float:
    """
    Weighted sum fusion of the two verification branches.

    fused = α * cnn_score  +  (1 - α) * minutiae_score

    Parameters
    ----------
    cnn_score      : similarity from the deep-learning branch  [0, 1]
    minutiae_score : matching score from the traditional branch [0, 1]
    alpha          : weight for CNN branch (default 0.5 = equal weighting)

    Returns
    -------
    fused_score : float in [0, 1]
    """
    return alpha * cnn_score + (1.0 - alpha) * minutiae_score


def verify(fused_score: float, threshold: float = 0.45) -> bool:
    """
    Final verification decision.

    Returns True (match) if fused_score >= threshold, else False (no match).
    """
    return fused_score >= threshold



# Convenience: full verification pipeline


def full_verification(image_path_1: str,
                      image_path_2: str,
                      model_path: str = 'finger_model.h5',
                      alpha: float = 0.5,
                      threshold: float = 0.45) -> dict:
    """
    End-to-end verification of two fingerprint images.

    Pipeline:
      1. Extract CNN embeddings → compute CNN similarity score
      2. Extract minutiae from both images → compute minutiae match score
      3. Fuse scores with weighted sum
      4. Apply threshold to make match / no-match decision

    Returns
    -------
    dict with keys:
      'cnn_score', 'minutiae_score', 'fused_score',
      'is_match', 'minutiae_1', 'minutiae_2'
    """
    # CNN branch
    try:
        emb1 = get_cnn_embedding(image_path_1, model_path)
        emb2 = get_cnn_embedding(image_path_2, model_path)
        cnn_sc = cnn_similarity_score(emb1, emb2)
    except Exception as e:
        print(f"[CNN Branch] Warning: {e}. Falling back to 0.0")
        cnn_sc = 0.0

    # Minutiae branch
    img1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not read one of the input images.")

    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))

    m1 = extract_minutiae(img1)
    m2 = extract_minutiae(img2)
    min_sc = minutiae_match_score(m1['all_minutiae'], m2['all_minutiae'])

    # Fusion
    fused = fuse_scores(cnn_sc, min_sc, alpha)
    decision = verify(fused, threshold)

    return {
        'cnn_score': round(cnn_sc, 4),
        'minutiae_score': round(min_sc, 4),
        'fused_score': round(fused, 4),
        'is_match': decision,
        'minutiae_1': m1,
        'minutiae_2': m2,
    }
