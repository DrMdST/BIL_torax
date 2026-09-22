import type { NRRDData } from './nrrd';
import { toGray, get2DSlice, normalizeToUint8 } from './imageOps';

export interface FeatureResult {
  [key: string]: number;
}

function mean(arr: Float32Array): number {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

function variance(arr: Float32Array, m?: number): number {
  const mu = m ?? mean(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) {
    const d = arr[i] - mu;
    s += d * d;
  }
  return s / arr.length;
}

function std(arr: Float32Array, m?: number): number {
  return Math.sqrt(variance(arr, m));
}

function skewness(arr: Float32Array, m?: number, s?: number): number {
  const mu = m ?? mean(arr);
  const sigma = s ?? std(arr, mu);
  if (sigma === 0) return 0;
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    sum += Math.pow((arr[i] - mu) / sigma, 3);
  }
  return sum / arr.length;
}

function kurtosis(arr: Float32Array, m?: number, s?: number): number {
  const mu = m ?? mean(arr);
  const sigma = s ?? std(arr, mu);
  if (sigma === 0) return 0;
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    sum += Math.pow((arr[i] - mu) / sigma, 4);
  }
  return sum / arr.length - 3;
}

function percentile(arr: Float32Array, p: number): number {
  const sorted = Array.from(arr).sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function entropy(arr: Float32Array): number {
  const hist = new Map<number, number>();
  for (let i = 0; i < arr.length; i++) {
    const v = Math.round(arr[i] * 100) / 100;
    hist.set(v, (hist.get(v) || 0) + 1);
  }
  let h = 0;
  const n = arr.length;
  for (const count of hist.values()) {
    const prob = count / n;
    if (prob > 0) h -= prob * Math.log2(prob);
  }
  return h;
}

function glcmFeatures(arr: Float32Array, width: number, height: number): {
  contrast: number; correlation: number; energy: number; homogeneity: number;
} {
  const levels = 8;
  const quantized = new Uint8Array(arr.length);
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < min) min = arr[i];
    if (arr[i] > max) max = arr[i];
  }
  const range = max - min || 1;
  for (let i = 0; i < arr.length; i++) {
    quantized[i] = Math.min(levels - 1, Math.floor(((arr[i] - min) / range) * (levels - 1)));
  }

  const glcm = new Float32Array(levels * levels);
  let count = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width - 1; x++) {
      const a = quantized[y * width + x];
      const b = quantized[y * width + x + 1];
      glcm[a * levels + b]++;
      glcm[b * levels + a]++;
      count += 2;
    }
  }

  if (count > 0) {
    for (let i = 0; i < glcm.length; i++) glcm[i] /= count;
  }

  let contrast = 0, energy = 0, homogeneity = 0, correlation = 0;
  let muI = 0, muJ = 0, sigmaI = 0, sigmaJ = 0;

  for (let i = 0; i < levels; i++) {
    for (let j = 0; j < levels; j++) {
      const p = glcm[i * levels + j];
      contrast += p * (i - j) * (i - j);
      energy += p * p;
      homogeneity += p / (1 + (i - j) * (i - j));
      muI += i * p;
      muJ += j * p;
    }
  }

  for (let i = 0; i < levels; i++) {
    for (let j = 0; j < levels; j++) {
      const p = glcm[i * levels + j];
      sigmaI += (i - muI) * (i - muI) * p;
      sigmaJ += (j - muJ) * (j - muJ) * p;
    }
  }

  sigmaI = Math.sqrt(sigmaI);
  sigmaJ = Math.sqrt(sigmaJ);

  if (sigmaI > 0 && sigmaJ > 0) {
    for (let i = 0; i < levels; i++) {
      for (let j = 0; j < levels; j++) {
        const p = glcm[i * levels + j];
        correlation += ((i - muI) * (j - muJ) * p) / (sigmaI * sigmaJ);
      }
    }
  }

  return { contrast, correlation, energy, homogeneity };
}

function morphologicalFeatures(maskSlice: Float32Array, width: number, height: number): {
  area: number; perimeter: number; eccentricity: number; extent: number;
} {
  let area = 0;
  let minR = height, maxR = 0, minC = width, maxC = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (maskSlice[y * width + x] > 0) {
        area++;
        if (y < minR) minR = y;
        if (y > maxR) maxR = y;
        if (x < minC) minC = x;
        if (x > maxC) maxC = x;
      }
    }
  }

  if (area === 0) return { area: 0, perimeter: 0, eccentricity: 0, extent: 0 };

  let perimeter = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (maskSlice[y * width + x] > 0) {
        let neighbors = 0;
        if (y > 0 && maskSlice[(y - 1) * width + x] > 0) neighbors++;
        if (y < height - 1 && maskSlice[(y + 1) * width + x] > 0) neighbors++;
        if (x > 0 && maskSlice[y * width + x - 1] > 0) neighbors++;
        if (x < width - 1 && maskSlice[y * width + x + 1] > 0) neighbors++;
        if (neighbors < 4) perimeter++;
      }
    }
  }

  const bboxW = maxC - minC + 1;
  const bboxH = maxR - minR + 1;
  const majorAxis = Math.max(bboxW, bboxH);
  const minorAxis = Math.min(bboxW, bboxH);
  const eccentricity = majorAxis === 0 ? 0 : Math.sqrt(1 - (minorAxis * minorAxis) / (majorAxis * majorAxis));
  const extent = area / (bboxW * bboxH);

  return { area, perimeter, eccentricity, extent };
}

export function extractFeatures(
  image: NRRDData,
  mask: NRRDData,
  sliceIndex: number = 0
): FeatureResult {
  const imageGray = toGray(image);
  const maskGray = toGray(mask);

  const imgSlice = get2DSlice(imageGray, image.shape, sliceIndex);
  const maskSlice = get2DSlice(maskGray, mask.shape, sliceIndex);

  const { slice: imgData, width, height } = imgSlice;
  const { slice: mskData } = maskSlice;

  const maskedValues: number[] = [];
  for (let i = 0; i < imgData.length; i++) {
    if (mskData[i] > 0) maskedValues.push(imgData[i]);
  }

  if (maskedValues.length === 0) {
    return {
      'Voxels_in_mask': 0,
      'Mean': 0, 'Std': 0, 'Variance': 0, 'Skewness': 0, 'Kurtosis': 0,
      'Min': 0, 'Max': 0, 'Range': 0, 'Median': 0,
      'P10': 0, 'P25': 0, 'P75': 0, 'P90': 0,
      'Entropy': 0,
      'GLCM_Contrast': 0, 'GLCM_Correlation': 0, 'GLCM_Energy': 0, 'GLCM_Homogeneity': 0,
      'Mask_Area': 0, 'Mask_Perimeter': 0, 'Mask_Eccentricity': 0, 'Mask_Extent': 0,
    };
  }

  const maskedArr = new Float32Array(maskedValues);
  const mu = mean(maskedArr);
  const sigma = std(maskedArr, mu);
  const skew = skewness(maskedArr, mu, sigma);
  const kurt = kurtosis(maskedArr, mu, sigma);
  const med = percentile(maskedArr, 50);
  const p10 = percentile(maskedArr, 10);
  const p25 = percentile(maskedArr, 25);
  const p75 = percentile(maskedArr, 75);
  const p90 = percentile(maskedArr, 90);
  const ent = entropy(maskedArr);
  const glcm = glcmFeatures(imgData, width, height);
  const morph = morphologicalFeatures(mskData, width, height);

  return {
    'Voxels_in_mask': maskedValues.length,
    'Mean': mu,
    'Std': sigma,
    'Variance': variance(maskedArr, mu),
    'Skewness': skew,
    'Kurtosis': kurt,
    'Min': Math.min(...maskedValues),
    'Max': Math.max(...maskedValues),
    'Range': Math.max(...maskedValues) - Math.min(...maskedValues),
    'Median': med,
    'P10': p10,
    'P25': p25,
    'P75': p75,
    'P90': p90,
    'Entropy': ent,
    'GLCM_Contrast': glcm.contrast,
    'GLCM_Correlation': glcm.correlation,
    'GLCM_Energy': glcm.energy,
    'GLCM_Homogeneity': glcm.homogeneity,
    'Mask_Area': morph.area,
    'Mask_Perimeter': morph.perimeter,
    'Mask_Eccentricity': morph.eccentricity,
    'Mask_Extent': morph.extent,
  };
}

export function extractAllSlices(
  image: NRRDData,
  mask: NRRDData,
  maxSlices: number = 50
): FeatureResult[] {
  const depth = Math.min(
    image.shape.length >= 3 ? image.shape[0] : 1,
    maxSlices
  );

  const results: FeatureResult[] = [];
  for (let i = 0; i < depth; i++) {
    const features = extractFeatures(image, mask, i);
    results.push({ ...features, 'Slice': i });
  }
  return results;
}
