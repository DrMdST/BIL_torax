import type { NRRDData } from './nrrd';

export function toGray(nrrd: NRRDData): Float32Array {
  const shape = nrrd.shape;
  const data = nrrd.data;

  if (shape.length === 2) {
    const result = new Float32Array(shape[0] * shape[1]);
    for (let i = 0; i < result.length; i++) result[i] = data[i];
    return result;
  }

  if (shape.length === 3) {
    if (shape[2] === 3 || shape[2] === 4) {
      const h = shape[0], w = shape[1];
      const result = new Float32Array(h * w);
      for (let i = 0; i < h * w; i++) {
        const r = data[i * 3], g = data[i * 3 + 1], b = data[i * 3 + 2];
        result[i] = 0.299 * r + 0.587 * g + 0.114 * b;
      }
      return result;
    }

    const total = shape.reduce((a, b) => a * b, 1);
    const result = new Float32Array(total);
    for (let i = 0; i < total; i++) result[i] = data[i];
    return result;
  }

  if (shape.length === 4 && (shape[3] === 3 || shape[3] === 4)) {
    const d = shape[0], h = shape[1], w = shape[2];
    const result = new Float32Array(d * h * w);
    for (let i = 0; i < d * h * w; i++) {
      const r = data[i * 3], g = data[i * 3 + 1], b = data[i * 3 + 2];
      result[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    }
    return result;
  }

  const total = shape.reduce((a, b) => a * b, 1);
  const result = new Float32Array(total);
  for (let i = 0; i < total; i++) result[i] = data[i];
  return result;
}

export function transpose3D(
  data: Float32Array,
  shape: number[],
  axes: [number, number, number]
): { data: Float32Array; shape: number[] } {
  if (shape.length !== 3) return { data, shape };

  const [d0, d1, d2] = shape;
  const [a0, a1, a2] = axes;
  const newShape = [shape[a0], shape[a1], shape[a2]];
  const result = new Float32Array(d0 * d1 * d2);

  for (let i = 0; i < d0; i++) {
    for (let j = 0; j < d1; j++) {
      for (let k = 0; k < d2; k++) {
        const srcIdx = i * d1 * d2 + j * d2 + k;
        const idx = [i, j, k];
        const dstIdx =
          idx[a0] * newShape[1] * newShape[2] +
          idx[a1] * newShape[2] +
          idx[a2];
        result[dstIdx] = data[srcIdx];
      }
    }
  }

  return { data: result, shape: newShape };
}

export function squeeze(data: Float32Array, shape: number[]): { data: Float32Array; shape: number[] } {
  const newShape: number[] = [];
  const strides: number[] = [];

  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    if (shape[i] !== 1) {
      newShape.unshift(shape[i]);
      strides.unshift(stride);
    }
    stride *= shape[i];
  }

  if (newShape.length === 0) {
    return { data: new Float32Array([data[0]]), shape: [1] };
  }

  return { data, shape: newShape };
}

export function get2DSlice(
  data: Float32Array,
  shape: number[],
  sliceIndex: number = 0
): { slice: Float32Array; width: number; height: number } {
  if (shape.length === 2) {
    return { slice: data, width: shape[1], height: shape[0] };
  }

  if (shape.length === 3) {
    const [d0, d1, d2] = shape;
    const idx = Math.min(sliceIndex, d0 - 1);
    const width = d2, height = d1;
    const slice = new Float32Array(width * height);
    const offset = idx * width * height;
    for (let i = 0; i < width * height; i++) {
      slice[i] = data[offset + i];
    }
    return { slice, width, height };
  }

  const total = shape.reduce((a, b) => a * b, 1);
  const flat = new Float32Array(total);
  for (let i = 0; i < total; i++) flat[i] = data[i];
  return { slice: flat, width: shape[shape.length - 1], height: shape[0] };
}

export function normalizeToUint8(data: Float32Array): Uint8Array {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
  }
  const range = max - min || 1;
  const result = new Uint8Array(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = Math.round(((data[i] - min) / range) * 255);
  }
  return result;
}
