"""
NeuroRad Analytics - Core Processing Module
============================================
Processes NRRD medical images and segmentation masks to extract
radiomic features using pyradiomics + SimpleITK.

Pipeline:
  1. Load .nrrd image and .seg.nrrd mask via SimpleITK
  2. For each slice:
     a. Apply the segmentation mask to define the region of interest (ROI)
     b. Compute first-order statistics ONLY from image voxels inside the mask
     c. Compute GLCM texture features ONLY from gray-level co-occurrences inside the mask
     d. Compute shape/morphological features from the mask geometry
  3. Export results to .xlsx
  4. Save processed images (image with mask overlay, mask alone) as .png

Key principle: ALL radiomic features are computed exclusively from the
region of interest defined by the .seg.nrrd mask. Voxels outside the mask
are never included in any calculation.
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


def load_nrrd_as_sitk(path: str) -> sitk.Image:
    """Load an NRRD file as a SimpleITK image."""
    return sitk.ReadImage(path)


def to_gray(data: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA data to grayscale using luminance weights."""
    if data.ndim == 2:
        return data.astype(np.float32)
    if data.ndim == 3 and data.shape[-1] in (3, 4):
        rgb = data[..., :3].astype(np.float32)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    if data.ndim == 4 and data.shape[-1] in (3, 4):
        rgb = data[..., :3].astype(np.float32)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return data.astype(np.float32).squeeze()


def normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    """Normalize intensity values to 0-255 uint8 range for visualization."""
    mn, mx = float(slice_data.min()), float(slice_data.max())
    if mx - mn < 1e-8:
        return np.zeros_like(slice_data, dtype=np.uint8)
    return ((slice_data - mn) / (mx - mn) * 255).astype(np.uint8)


def get_2d_slice(data: np.ndarray, index: int = 0) -> np.ndarray:
    """Extract a 2D slice from a 3D volume at the given index."""
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        idx = min(index, data.shape[0] - 1)
        return data[idx]
    return data.squeeze()


def compute_first_order(masked_values: np.ndarray) -> Dict[str, float]:
    """Compute first-order statistics from image intensities WITHIN the mask only."""
    if masked_values.size == 0:
        return {k: 0.0 for k in [
            'Mean', 'Std', 'Variance', 'Skewness', 'Kurtosis',
            'Min', 'Max', 'Range', 'Median',
            'P10', 'P25', 'P75', 'P90', 'Entropy'
        ]}

    mu = float(masked_values.mean())
    sigma = float(masked_values.std())

    if sigma > 1e-8:
        skew = float(((masked_values - mu) / sigma) ** 3).mean())
        kurt = float(((masked_values - mu) / sigma) ** 4).mean() - 3.0)
    else:
        skew = 0.0
        kurt = 0.0

    hist, _ = np.histogram(masked_values, bins=32)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    ent = float(-(hist * np.log2(hist)).sum())

    return {
        'Mean': mu,
        'Std': sigma,
        'Variance': float(masked_values.var()),
        'Skewness': skew,
        'Kurtosis': kurt,
        'Min': float(masked_values.min()),
        'Max': float(masked_values.max()),
        'Range': float(masked_values.max() - masked_values.min()),
        'Median': float(np.median(masked_values)),
        'P10': float(np.percentile(masked_values, 10)),
        'P25': float(np.percentile(masked_values, 25)),
        'P75': float(np.percentile(masked_values, 75)),
        'P90': float(np.percentile(masked_values, 90)),
        'Entropy': ent,
    }


def compute_glcm_features_masked(img_slice: np.ndarray, mask_slice: np.ndarray, levels: int = 8) -> Dict[str, float]:
    """
    Compute GLCM texture features ONLY from pixels inside the mask.

    Only pixel pairs where BOTH pixels are inside the mask contribute to the
    gray-level co-occurrence matrix. This ensures texture features reflect only
    the region of interest.
    """
    binary = mask_slice > 0
    if binary.sum() == 0:
        return {k: 0.0 for k in ['GLCM_Contrast', 'GLCM_Correlation', 'GLCM_Energy', 'GLCM_Homogeneity']}

    # Quantize image to 'levels' gray levels using the range of the MASKED region only
    masked_vals = img_slice[binary]
    mn, mx = float(masked_vals.min()), float(masked_vals.max())
    rng = mx - mn if mx > mn else 1.0
    quantized = np.clip(((img_slice - mn) / rng * (levels - 1)).astype(np.int32), 0, levels - 1)

    glcm = np.zeros((levels, levels), dtype=np.float64)
    h, w = quantized.shape
    count = 0

    # Only count co-occurrences where BOTH pixels are inside the mask
    for y in range(h):
        for x in range(w - 1):
            if binary[y, x] and binary[y, x + 1]:
                a, b = quantized[y, x], quantized[y, x + 1]
                glcm[a, b] += 1
                glcm[b, a] += 1
                count += 2

    if count > 0:
        glcm /= count
    else:
        return {k: 0.0 for k in ['GLCM_Contrast', 'GLCM_Correlation', 'GLCM_Energy', 'GLCM_Homogeneity']}

    contrast, energy, homogeneity = 0.0, 0.0, 0.0
    mu_i, mu_j = 0.0, 0.0
    for i in range(levels):
        for j in range(levels):
            p = glcm[i, j]
            contrast += p * (i - j) ** 2
            energy += p * p
            homogeneity += p / (1 + (i - j) ** 2)
            mu_i += i * p
            mu_j += j * p

    sig_i, sig_j = 0.0, 0.0
    for i in range(levels):
        for j in range(levels):
            p = glcm[i, j]
            sig_i += (i - mu_i) ** 2 * p
            sig_j += (j - mu_j) ** 2 * p
    sig_i = np.sqrt(sig_i)
    sig_j = np.sqrt(sig_j)

    correlation = 0.0
    if sig_i > 0 and sig_j > 0:
        for i in range(levels):
            for j in range(levels):
                p = glcm[i, j]
                correlation += (i - mu_i) * (j - mu_j) * p / (sig_i * sig_j)

    return {
        'GLCM_Contrast': float(contrast),
        'GLCM_Correlation': float(correlation),
        'GLCM_Energy': float(energy),
        'GLCM_Homogeneity': float(homogeneity),
    }


def compute_morphology(mask_slice: np.ndarray) -> Dict[str, float]:
    """Compute morphological/shape features from the binary segmentation mask."""
    binary = mask_slice > 0
    area = int(binary.sum())
    if area == 0:
        return {'Mask_Area': 0, 'Mask_Perimeter': 0, 'Mask_Eccentricity': 0, 'Mask_Extent': 0}

    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox_w = cmax - cmin + 1
    bbox_h = rmax - rmin + 1

    perimeter = 0
    h, w = binary.shape
    for y in range(h):
        for x in range(w):
            if binary[y, x]:
                neighbors = 0
                if y > 0 and binary[y - 1, x]: neighbors += 1
                if y < h - 1 and binary[y + 1, x]: neighbors += 1
                if x > 0 and binary[y, x - 1]: neighbors += 1
                if x < w - 1 and binary[y, x + 1]: neighbors += 1
                if neighbors < 4:
                    perimeter += 1

    major = max(bbox_w, bbox_h)
    minor = min(bbox_w, bbox_h)
    eccentricity = float(np.sqrt(1 - (minor / major) ** 2)) if major > 0 else 0.0
    extent = area / (bbox_w * bbox_h) if bbox_w * bbox_h > 0 else 0.0

    return {
        'Mask_Area': area,
        'Mask_Perimeter': perimeter,
        'Mask_Eccentricity': eccentricity,
        'Mask_Extent': extent,
    }


def extract_features_slice(img_slice: np.ndarray, mask_slice: np.ndarray) -> Dict[str, float]:
    """
    Extract ALL radiomic features for a single 2D slice.

    The mask defines the region of interest (ROI). Only image voxels where the
    mask is non-zero are used for first-order and GLCM calculations.
    Shape features are derived from the mask geometry itself.
    """
    binary = mask_slice > 0
    masked_values = img_slice[binary]

    first_order = compute_first_order(masked_values)
    glcm = compute_glcm_features_masked(img_slice, mask_slice)
    morph = compute_morphology(mask_slice)

    return {
        'Voxels_in_mask': int(masked_values.size),
        **first_order,
        **glcm,
        **morph,
    }


def extract_features(image_path: str, mask_path: str, max_slices: int = 50) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """
    Main entry point: load NRRD files, extract features for all slices.

    Args:
        image_path: Path to the .nrrd image file
        mask_path: Path to the .seg.nrrd segmentation mask file
        max_slices: Maximum number of slices to process

    Returns:
        results: List of feature dictionaries (one per slice)
        processed_image: 2D numpy array (middle slice of image)
        processed_mask: 2D numpy array (middle slice of mask)
    """
    # Load via SimpleITK for proper NRRD handling
    image_sitk = sitk.ReadImage(image_path)
    mask_sitk = sitk.ReadImage(mask_path)

    image_data = sitk.GetArrayFromImage(image_sitk)
    mask_data = sitk.GetArrayFromImage(mask_sitk)

    img_gray = to_gray(image_data)
    mask_gray = to_gray(mask_data)

    depth = min(img_gray.shape[0] if img_gray.ndim >= 3 else 1, max_slices)

    results: List[Dict] = []
    for i in range(depth):
        img_slice = get_2d_slice(img_gray, i)
        mask_slice = get_2d_slice(mask_gray, i)

        if img_slice.shape != mask_slice.shape:
            mask_slice = mask_slice[:img_slice.shape[0], :img_slice.shape[1]]

        features = extract_features_slice(img_slice, mask_slice)
        features['Slice'] = i
        results.append(features)

    mid = depth // 2
    processed_image = get_2d_slice(img_gray, mid)
    processed_mask = get_2d_slice(mask_gray, mid)

    return results, processed_image, processed_mask


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python test.py <image.nrrd> <mask.seg.nrrd> [output_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    mask_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else '.'

    results, proc_img, proc_mask = extract_features(image_path, mask_path)

    # Export to Excel
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Features"
    headers = list(results[0].keys())
    for col, h in enumerate(headers, 1):
        ws.cell(row=1, column=col, value=h)
    for row_idx, row_data in enumerate(results, 2):
        for col_idx, key in enumerate(headers, 1):
            ws.cell(row=row_idx, column=col_idx, value=row_data.get(key, 0))
    xlsx_path = os.path.join(output_dir, "radiomics_results.xlsx")
    wb.save(xlsx_path)

    # Save processed images
    from PIL import Image
    img_norm = normalize_slice(proc_img)
    Image.fromarray(np.stack([img_norm] * 3, axis=-1)).save(os.path.join(output_dir, "processed_image.png"))
    mask_norm = normalize_slice(proc_mask)
    Image.fromarray(np.stack([mask_norm] * 3, axis=-1)).save(os.path.join(output_dir, "processed_mask.png"))

    print(f"Done! {len(results)} slices processed.")
    print(f"  Excel: {xlsx_path}")
    print(f"  Image: {os.path.join(output_dir, 'processed_image.png')}")
    print(f"  Mask:  {os.path.join(output_dir, 'processed_mask.png')}")
