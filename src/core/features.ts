import type { NRRDData } from './nrrd';
import { squeezeAndTranspose, getSlice, getSliceDimensions } from './nrrd';
import { toGray } from './imageOps';

export interface FeatureResult {
  [key: string]: number;
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

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

function uniformity(arr: Float32Array): number {
  const hist = new Map<number, number>();
  for (let i = 0; i < arr.length; i++) {
    const v = Math.round(arr[i] * 100) / 100;
    hist.set(v, (hist.get(v) || 0) + 1);
  }
  let s = 0;
  const n = arr.length;
  for (const count of hist.values()) {
    const p = count / n;
    s += p * p;
  }
  return s;
}

function rms(arr: Float32Array, m?: number): number {
  const mu = m ?? mean(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) {
    s += (arr[i] - mu) * (arr[i] - mu);
  }
  return Math.sqrt(s / arr.length);
}

function mad(arr: Float32Array, m?: number): number {
  const mu = m ?? mean(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) {
    s += Math.abs(arr[i] - mu);
  }
  return s / arr.length;
}

// ---------------------------------------------------------------------------
// First-order features (18 features, matching pyradiomics firstorder)
// ---------------------------------------------------------------------------

function computeFirstOrder(maskedValues: Float32Array): FeatureResult {
  if (maskedValues.length === 0) {
    const keys = [
      'FirstOrder_Mean', 'FirstOrder_StandardDeviation', 'FirstOrder_Variance',
      'FirstOrder_Skewness', 'FirstOrder_Kurtosis', 'FirstOrder_Median',
      'FirstOrder_Minimum', 'FirstOrder_Maximum', 'FirstOrder_Range',
      'FirstOrder_10Percentile', 'FirstOrder_25Percentile', 'FirstOrder_75Percentile',
      'FirstOrder_90Percentile', 'FirstOrder_Entropy', 'FirstOrder_Uniformity',
      'FirstOrder_RootMeanSquared', 'FirstOrder_MedianAbsoluteDeviation',
      'FirstOrder_Energy',
    ];
    return Object.fromEntries(keys.map(k => [k, 0]));
  }

  const mu = mean(maskedValues);
  const sigma = std(maskedValues, mu);
  let energy = 0;
  for (let i = 0; i < maskedValues.length; i++) energy += maskedValues[i] * maskedValues[i];

  return {
    'FirstOrder_Mean': mu,
    'FirstOrder_StandardDeviation': sigma,
    'FirstOrder_Variance': variance(maskedValues, mu),
    'FirstOrder_Skewness': skewness(maskedValues, mu, sigma),
    'FirstOrder_Kurtosis': kurtosis(maskedValues, mu, sigma),
    'FirstOrder_Median': percentile(maskedValues, 50),
    'FirstOrder_Minimum': Math.min(...maskedValues),
    'FirstOrder_Maximum': Math.max(...maskedValues),
    'FirstOrder_Range': Math.max(...maskedValues) - Math.min(...maskedValues),
    'FirstOrder_10Percentile': percentile(maskedValues, 10),
    'FirstOrder_25Percentile': percentile(maskedValues, 25),
    'FirstOrder_75Percentile': percentile(maskedValues, 75),
    'FirstOrder_90Percentile': percentile(maskedValues, 90),
    'FirstOrder_Entropy': entropy(maskedValues),
    'FirstOrder_Uniformity': uniformity(maskedValues),
    'FirstOrder_RootMeanSquared': rms(maskedValues, mu),
    'FirstOrder_MedianAbsoluteDeviation': mad(maskedValues, mu),
    'FirstOrder_Energy': energy,
  };
}

// ---------------------------------------------------------------------------
// GLCM features (24 features, 4 angles, matching pyradiomics glcm)
// Uses binWidth=25 quantization (fixed bin width, not fixed levels)
// ---------------------------------------------------------------------------

const GLCM_ANGLES = [
  [0, 1] as const,     // 0 degrees
  [1, 1] as const,     // 45 degrees
  [1, 0] as const,     // 90 degrees
  [1, -1] as const,    // 135 degrees
];

function quantizeWithBinWidth(imgData: Float32Array, maskData: Float32Array, binWidth: number = 25): { quantized: Int32Array, min: number, numLevels: number } {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < imgData.length; i++) {
    if (maskData[i] > 0) {
      if (imgData[i] < min) min = imgData[i];
      if (imgData[i] > max) max = imgData[i];
    }
  }
  if (min === Infinity) min = 0;
  const numLevels = Math.max(1, Math.ceil((max - min) / binWidth) + 1);
  const quantized = new Int32Array(imgData.length);
  for (let i = 0; i < imgData.length; i++) {
    quantized[i] = Math.min(numLevels - 1, Math.max(0, Math.floor((imgData[i] - min) / binWidth)));
  }
  return { quantized, min, numLevels };
}

function computeGLCMAngle(
  quantized: Int32Array,
  maskData: Float32Array,
  width: number,
  height: number,
  dy: number,
  dx: number,
  numLevels: number
): Float32Array {
  const glcm = new Float32Array(numLevels * numLevels);
  let count = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const ny = y + dy;
      const nx = x + dx;
      if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
      const idxA = y * width + x;
      const idxB = ny * width + nx;
      if (maskData[idxA] > 0 && maskData[idxB] > 0) {
        const a = quantized[idxA];
        const b = quantized[idxB];
        glcm[a * numLevels + b]++;
        glcm[b * numLevels + a]++;
        count += 2;
      }
    }
  }

  if (count > 0) {
    for (let i = 0; i < glcm.length; i++) glcm[i] /= count;
  }
  return glcm;
}

function glcmFeaturesFromMatrix(glcm: Float32Array, numLevels: number): FeatureResult {
  let contrast = 0, correlation = 0, energy = 0, homogeneity = 0;
  let clusterShade = 0, clusterProminence = 0, clusterTendency = 0;
  let differenceAverage = 0, differenceVariance = 0, differenceEntropy = 0;
  let sumAverage = 0, sumVariance = 0, sumEntropy = 0;
  let muI = 0, muJ = 0, idmn = 0, idn = 0, imc1 = 0, imc2 = 0;
  let maxProb = 0;

  for (let i = 0; i < numLevels; i++) {
    for (let j = 0; j < numLevels; j++) {
      const p = glcm[i * numLevels + j];
      if (p > maxProb) maxProb = p;
      const diff = i - j;
      const sum = i + j;
      contrast += p * diff * diff;
      energy += p * p;
      homogeneity += p / (1 + diff * diff);
      muI += i * p;
      muJ += j * p;
      sumAverage += sum * p;
      idmn += p / (1 + diff * diff / (numLevels * numLevels));
      idn += p / (1 + Math.abs(diff) / numLevels);
      differenceAverage += Math.abs(diff) * p;
    }
  }

  let sigI = 0, sigJ = 0;
  for (let i = 0; i < numLevels; i++) {
    for (let j = 0; j < numLevels; j++) {
      const p = glcm[i * numLevels + j];
      sigI += (i - muI) * (i - muI) * p;
      sigJ += (j - muJ) * (j - muJ) * p;
    }
  }
  sigI = Math.sqrt(sigI);
  sigJ = Math.sqrt(sigJ);

  if (sigI > 0 && sigJ > 0) {
    for (let i = 0; i < numLevels; i++) {
      for (let j = 0; j < numLevels; j++) {
        const p = glcm[i * numLevels + j];
        correlation += (i - muI) * (j - muJ) * p / (sigI * sigJ);
      }
    }
  }

  for (let i = 0; i < numLevels; i++) {
    for (let j = 0; j < numLevels; j++) {
      const p = glcm[i * numLevels + j];
      const diff = i - j;
      const sum = i + j;
      clusterShade += Math.pow(sum - 2 * muI, 3) * p;
      clusterProminence += Math.pow(sum - 2 * muI, 4) * p;
      clusterTendency += Math.pow(sum - 2 * muI, 2) * p;
      differenceVariance += Math.pow(Math.abs(diff) - differenceAverage, 2) * p;
      sumVariance += Math.pow(sum - sumAverage, 2) * p;
    }
  }

  // Entropy of difference and sum
  let diffEntropy = 0, sumEnt = 0;
  const diffHist = new Float32Array(numLevels);
  const sumHist = new Float32Array(2 * numLevels - 1);
  for (let i = 0; i < numLevels; i++) {
    for (let j = 0; j < numLevels; j++) {
      const p = glcm[i * numLevels + j];
      diffHist[Math.abs(i - j)] += p;
      sumHist[i + j] += p;
    }
  }
  for (let i = 0; i < numLevels; i++) {
    if (diffHist[i] > 0) diffEntropy -= diffHist[i] * Math.log2(diffHist[i]);
  }
  for (let i = 0; i < 2 * numLevels - 1; i++) {
    if (sumHist[i] > 0) sumEntropy -= sumHist[i] * Math.log2(sumHist[i]);
  }

  // IMC1 and IMC2
  let hx = 0, hy = 0, hxy = 0, hxy1 = 0, hxy2 = 0;
  const px = new Float32Array(numLevels);
  const py = new Float32Array(numLevels);
  for (let i = 0; i < numLevels; i++) {
    for (let j = 0; j < numLevels; j++) {
      const p = glcm[i * numLevels + j];
      px[i] += p;
      py[j] += p;
      if (p > 0) hxy -= p * Math.log2(p);
    }
  }
  for (let i = 0; i < numLevels; i++) {
    if (px[i] > 0) hx -= px[i] * Math.log2(px[i]);
    if (py[i] > 0) hy -= py[i] * Math.log2(py[i]);
  }
  for (let i = 0; i < numLevels; i++) {
    for (let j = 0; j < numLevels; j++) {
      const p = glcm[i * numLevels + j];
      if (px[i] > 0 && py[j] > 0) {
        const pxy = px[i] * py[j];
        if (p > 0) hxy1 -= p * Math.log2(pxy);
        if (pxy > 0) hxy2 -= pxy * Math.log2(pxy);
      }
    }
  }
  imc1 = -hxy + hxy1;
  imc2 = Math.sqrt(Math.max(0, 1 - Math.exp(-2 * (hxy - hxy2))));

  return {
    'GLCM_Contrast': contrast,
    'GLCM_Correlation': correlation,
    'GLCM_JointEnergy': energy,
    'GLCM_Idm': homogeneity,
    'GLCM_ClusterShade': clusterShade,
    'GLCM_ClusterProminence': clusterProminence,
    'GLCM_ClusterTendency': clusterTendency,
    'GLCM_DifferenceAverage': differenceAverage,
    'GLCM_DifferenceVariance': differenceVariance,
    'GLCM_DifferenceEntropy': diffEntropy,
    'GLCM_SumAverage': sumAverage,
    'GLCM_SumVariance': sumVariance,
    'GLCM_SumEntropy': sumEnt,
    'GLCM_Idmn': idmn,
    'GLCM_Idn': idn,
    'GLCM_Imc1': imc1,
    'GLCM_Imc2': imc2,
    'GLCM_MaximumProbability': maxProb,
    'GLCM_Autocorrelation': muI * muJ,
    'GLCM_JointVariance': sigI * sigI + sigJ * sigJ,
    'GLCM_JointEntropy': hxy,
    'GLCM_DifferenceEntropy': diffEntropy,
    'GLCM_SumEntropy': sumEnt,
    'GLCM_InverseVariance': contrast > 0 ? 1 / contrast : 0,
  };
}

function computeGLCM(imgData: Float32Array, maskData: Float32Array, width: number, height: number): FeatureResult {
  const { quantized, numLevels } = quantizeWithBinWidth(imgData, maskData, 25);

  const angleResults: FeatureResult[] = [];
  for (const [dy, dx] of GLCM_ANGLES) {
    const glcm = computeGLCMAngle(quantized, maskData, width, height, dy, dx, numLevels);
    angleResults.push(glcmFeaturesFromMatrix(glcm, numLevels));
  }

  // Average across angles
  const result: FeatureResult = {};
  const keys = Object.keys(angleResults[0]);
  for (const key of keys) {
    let sum = 0;
    for (const r of angleResults) sum += r[key];
    result[key] = sum / angleResults.length;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Shape features (2D, matching pyradiomics shape2D)
// ---------------------------------------------------------------------------

function computeShape(maskSlice: Float32Array, width: number, height: number): FeatureResult {
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

  if (area === 0) {
    return {
      'Shape_PixelArea': 0, 'Shape_Perimeter': 0, 'Shape_Eccentricity': 0,
      'Shape_Extent': 0, 'Shape_MajorAxisLength': 0, 'Shape_MinorAxisLength': 0,
      'Shape_Elongation': 0, 'Shape_Flatness': 0, 'Shape_Diameter': 0,
      'Shape_SurfaceArea': 0, 'Shape_Maximum2DDiameter': 0,
      'Shape_Minimum2DDiameter': 0, 'Shape_BoundingBoxArea': 0,
      'Shape_PerimeterSurfaceRatio': 0,
    };
  }

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
  const elongation = minorAxis > 0 ? majorAxis / minorAxis : 0;
  const flatness = majorAxis > 0 ? minorAxis / majorAxis : 0;
  const maxDiameter = Math.sqrt(bboxW * bboxW + bboxH * bboxH);
  const minDiameter = minorAxis;
  const surfaceArea = perimeter;
  const bboxArea = bboxW * bboxH;
  const periSurfRatio = area > 0 ? perimeter / area : 0;

  return {
    'Shape_PixelArea': area,
    'Shape_Perimeter': perimeter,
    'Shape_Eccentricity': eccentricity,
    'Shape_Extent': extent,
    'Shape_MajorAxisLength': majorAxis,
    'Shape_MinorAxisLength': minorAxis,
    'Shape_Elongation': elongation,
    'Shape_Flatness': flatness,
    'Shape_Diameter': Math.sqrt(area / Math.PI) * 2,
    'Shape_SurfaceArea': surfaceArea,
    'Shape_Maximum2DDiameter': maxDiameter,
    'Shape_Minimum2DDiameter': minDiameter,
    'Shape_BoundingBoxArea': bboxArea,
    'Shape_PerimeterSurfaceRatio': periSurfRatio,
  };
}

// ---------------------------------------------------------------------------
// Main extraction
// ---------------------------------------------------------------------------

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

  // Extract ONLY the image values within the mask region
  const maskedValuesList: number[] = [];
  for (let i = 0; i < imgData.length; i++) {
    if (mskData[i] > 0) maskedValuesList.push(imgData[i]);
  }

  if (maskedValuesList.length === 0) {
    const empty: FeatureResult = { 'Voxels_in_mask': 0 };
    const fo = computeFirstOrder(new Float32Array(0));
    const glcm = computeGLCM(imgData, mskData, width, height);
    const shape = computeShape(mskData, width, height);
    return { ...empty, ...fo, ...glcm, ...shape };
  }

  const maskedArr = new Float32Array(maskedValuesList);
  const firstOrder = computeFirstOrder(maskedArr);
  const glcm = computeGLCM(imgData, mskData, width, height);
  const shape = computeShape(mskData, width, height);

  return {
    'Voxels_in_mask': maskedValuesList.length,
    ...firstOrder,
    ...glcm,
    ...shape,
  };
}

export function extractAllSlices(
  image: NRRDData,
  mask: NRRDData,
  maxSlices: number = 50
): FeatureResult[] {
  const { depth: fullDepth } = getSliceDimensions(image, 0);
  const depth = Math.min(fullDepth, maxSlices);

  const results: FeatureResult[] = [];
  for (let i = 0; i < depth; i++) {
    const features = extractFeatures(image, mask, i);
    results.push({ ...features, 'Slice': i });
  }
  return results;
}
