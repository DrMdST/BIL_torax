export interface NRRDData {
  data: Int8Array | Uint8Array | Uint16Array | Uint32Array | Int16Array | Int32Array | Float32Array | Float64Array;
  shape: number[];
  spacing: number[];
  sizes: number[];
  endian: string;
  encoding: string;
  dtype: string;
}

type TypedArrayConstructor =
  | typeof Int8Array | typeof Uint8Array | typeof Int16Array
  | typeof Uint16Array | typeof Int32Array | typeof Uint32Array
  | typeof Float32Array | typeof Float64Array;

const DTYPE_MAP: Record<string, { arrayType: TypedArrayConstructor; size: number }> = {
  'int8': { arrayType: Int8Array, size: 1 },
  'int8_t': { arrayType: Int8Array, size: 1 },
  'signed char': { arrayType: Int8Array, size: 1 },
  'char': { arrayType: Int8Array, size: 1 },
  'uint8': { arrayType: Uint8Array, size: 1 },
  'uint8_t': { arrayType: Uint8Array, size: 1 },
  'unsigned char': { arrayType: Uint8Array, size: 1 },
  'uchar': { arrayType: Uint8Array, size: 1 },
  'int16': { arrayType: Int16Array, size: 2 },
  'int16_t': { arrayType: Int16Array, size: 2 },
  'short': { arrayType: Int16Array, size: 2 },
  'short int': { arrayType: Int16Array, size: 2 },
  'signed short': { arrayType: Int16Array, size: 2 },
  'signed short int': { arrayType: Int16Array, size: 2 },
  'uint16': { arrayType: Uint16Array, size: 2 },
  'uint16_t': { arrayType: Uint16Array, size: 2 },
  'unsigned short': { arrayType: Uint16Array, size: 2 },
  'unsigned short int': { arrayType: Uint16Array, size: 2 },
  'ushort': { arrayType: Uint16Array, size: 2 },
  'int32': { arrayType: Int32Array, size: 4 },
  'int32_t': { arrayType: Int32Array, size: 4 },
  'int': { arrayType: Int32Array, size: 4 },
  'signed int': { arrayType: Int32Array, size: 4 },
  'long': { arrayType: Int32Array, size: 4 },
  'long int': { arrayType: Int32Array, size: 4 },
  'longlong': { arrayType: Int32Array, size: 4 },
  'long long': { arrayType: Int32Array, size: 4 },
  'long long int': { arrayType: Int32Array, size: 4 },
  'uint32': { arrayType: Uint32Array, size: 4 },
  'uint32_t': { arrayType: Uint32Array, size: 4 },
  'unsigned': { arrayType: Uint32Array, size: 4 },
  'unsigned int': { arrayType: Uint32Array, size: 4 },
  'ulong': { arrayType: Uint32Array, size: 4 },
  'unsigned long': { arrayType: Uint32Array, size: 4 },
  'unsigned long int': { arrayType: Uint32Array, size: 4 },
  'ulonglong': { arrayType: Uint32Array, size: 4 },
  'unsigned long long': { arrayType: Uint32Array, size: 4 },
  'unsigned long long int': { arrayType: Uint32Array, size: 4 },
  'uint64': { arrayType: Uint32Array, size: 4 },
  'uint64_t': { arrayType: Uint32Array, size: 4 },
  'float32': { arrayType: Float32Array, size: 4 },
  'float': { arrayType: Float32Array, size: 4 },
  'double': { arrayType: Float64Array, size: 8 },
  'float64': { arrayType: Float64Array, size: 8 },
  'long double': { arrayType: Float64Array, size: 8 },
  'double float': { arrayType: Float64Array, size: 8 },
  'block': { arrayType: Uint8Array, size: 1 },
};

function parseHeader(headerText: string): Record<string, unknown> {
  const fields: Record<string, unknown> = {};
  const lines = headerText.split(/\r?\n/);

  for (const line of lines) {
    if (line.startsWith('NRRD') || line.startsWith('#') || line.trim() === '') continue;

    const colonIdx = line.indexOf(':');
    if (colonIdx === -1) continue;

    const key = line.substring(0, colonIdx).trim().toLowerCase();
    const value = line.substring(colonIdx + 1).trim();

    if (key === 'sizes' || key === 'spacing' || key === 'space directions' || key === 'sizes') {
      fields[key] = value.split(/\s+/).map(Number);
    } else if (key === 'dimension') {
      fields[key] = parseInt(value, 10);
    } else {
      fields[key] = value;
    }
  }

  return fields;
}

export async function parseNRRD(file: File): Promise<NRRDData> {
  const arrayBuffer = await file.arrayBuffer();
  const bytes = new Uint8Array(arrayBuffer);

  let headerEnd = -1;
  for (let i = 0; i < bytes.length - 4; i++) {
    if (bytes[i] === 10 && bytes[i + 1] === 10) {
      headerEnd = i + 2;
      break;
    }
  }

  if (headerEnd === -1) {
    for (let i = 0; i < bytes.length - 1; i++) {
      if (bytes[i] === 10) {
        headerEnd = i + 1;
        break;
      }
    }
  }

  if (headerEnd === -1) throw new Error('Invalid NRRD: could not find header end');

  const headerText = new TextDecoder('ascii').decode(bytes.subarray(0, headerEnd));
  const fields = parseHeader(headerText);

  const dtype = ((fields['type'] as string) || 'int16').toLowerCase().trim();
  const encoding = (fields['encoding'] as string) || 'raw';
  const endian = (fields['endian'] as string) || 'little';
  const shape = (fields['sizes'] as number[]) || [];
  const spacing = (fields['spacing'] as number[]) || (fields['space directions'] as number[]) || [];

  const dtInfo = DTYPE_MAP[dtype];
  if (!dtInfo) throw new Error(`Unsupported NRRD dtype: ${dtype}`);

  let totalElements = 1;
  for (const s of shape) totalElements *= s;

  const dataBytes = bytes.subarray(headerEnd);
  let rawData: ArrayBuffer;

  if (encoding === 'raw') {
    rawData = dataBytes.buffer.slice(dataBytes.byteOffset, dataBytes.byteOffset + dataBytes.byteLength);
  } else if (encoding === 'gzip' || encoding === 'gz') {
    const ds = new DecompressionStream('gzip');
    const decompressed = new Response(dataBytes).body!.pipeThrough(ds);
    const buf = await new Response(decompressed).arrayBuffer();
    rawData = buf;
  } else if (encoding === 'ascii') {
    const text = new TextDecoder('ascii').decode(dataBytes);
    const values = text.trim().split(/\s+/).map(Number);
    const tmp = new dtInfo.arrayType(values.length);
    for (let i = 0; i < values.length; i++) tmp[i] = values[i];
    return { data: tmp, shape, spacing, sizes: shape, endian, encoding, dtype };
  } else {
    throw new Error(`Unsupported NRRD encoding: ${encoding}`);
  }

  const typedArray = new dtInfo.arrayType(rawData, 0, totalElements);

  if (dtInfo.size > 1 && endian === 'big') {
    const view = new DataView(rawData);
    for (let i = 0; i < totalElements; i++) {
      const bs = i * dtInfo.size;
      if (dtInfo.size === 2) {
        const val = view.getUint16(bs, false);
        view.setUint16(bs, ((val & 0xff) << 8) | ((val >> 8) & 0xff), false);
      } else if (dtInfo.size === 4) {
        const val = view.getUint32(bs, false);
        view.setUint32(bs, ((val & 0xff) << 24) | ((val & 0xff00) << 8) | ((val >> 8) & 0xff00) | ((val >> 24) & 0xff), false);
      }
    }
  }

  return { data: typedArray, shape, spacing, sizes: shape, endian, encoding, dtype };
}

export function getSlice(nrrd: NRRDData, sliceIndex: number, axis: number = 0): Float32Array {
  const shape = nrrd.shape;
  const data = nrrd.data;

  if (shape.length === 2) {
    const w = shape[1], h = shape[0];
    const slice = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) slice[i] = data[i];
    return slice;
  }

  if (shape.length < 3) {
    const total = shape.reduce((a, b) => a * b, 1);
    const slice = new Float32Array(total);
    for (let i = 0; i < total; i++) slice[i] = data[i];
    return slice;
  }

  const [d0, d1, d2] = shape;
  let width: number, height: number, depth: number;

  if (axis === 0) { width = d2; height = d1; depth = d0; }
  else if (axis === 1) { width = d2; height = d0; depth = d1; }
  else { width = d1; height = d0; depth = d2; }

  if (sliceIndex < 0 || sliceIndex >= depth) throw new Error(`Slice ${sliceIndex} out of range (depth ${depth})`);

  const slice = new Float32Array(width * height);

  if (axis === 0) {
    const offset = sliceIndex * d1 * d2;
    for (let i = 0; i < width * height; i++) {
      slice[i] = data[offset + i];
    }
  } else if (axis === 1) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        slice[y * width + x] = data[y * d1 * d2 + sliceIndex * d2 + x];
      }
    }
  } else {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        slice[y * width + x] = data[y * d1 * d2 + x * d2 + sliceIndex];
      }
    }
  }

  return slice;
}

export function getSliceDimensions(nrrd: NRRDData, axis: number = 0): { width: number; height: number; depth: number } {
  const shape = nrrd.shape;
  if (shape.length === 2) return { width: shape[1], height: shape[0], depth: 1 };
  if (shape.length < 3) {
    const total = shape.reduce((a, b) => a * b, 1);
    return { width: total, height: 1, depth: 1 };
  }
  const [d0, d1, d2] = shape;
  if (axis === 0) return { width: d2, height: d1, depth: d0 };
  if (axis === 1) return { width: d2, height: d0, depth: d1 };
  return { width: d1, height: d0, depth: d2 };
}

export function normalizeSlice(slice: Float32Array): Uint8Array {
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < slice.length; i++) {
    if (slice[i] < min) min = slice[i];
    if (slice[i] > max) max = slice[i];
  }
  const range = max - min || 1;
  const result = new Uint8Array(slice.length);
  for (let i = 0; i < slice.length; i++) {
    result[i] = Math.round(((slice[i] - min) / range) * 255);
  }
  return result;
}

export function sliceToImageData(
  slice: Float32Array,
  width: number,
  height: number,
  maskSlice?: Float32Array
): ImageData {
  const normalized = normalizeSlice(slice);
  const imageData = new ImageData(width, height);

  if (maskSlice) {
    for (let i = 0; i < normalized.length; i++) {
      const px = i * 4;
      if (maskSlice[i] > 0) {
        imageData.data[px] = Math.round(normalized[i] * 0.5 + 128);
        imageData.data[px + 1] = normalized[i];
        imageData.data[px + 2] = Math.round(normalized[i] * 0.3);
      } else {
        imageData.data[px] = normalized[i];
        imageData.data[px + 1] = normalized[i];
        imageData.data[px + 2] = normalized[i];
      }
      imageData.data[px + 3] = 255;
    }
  } else {
    for (let i = 0; i < normalized.length; i++) {
      const px = i * 4;
      imageData.data[px] = normalized[i];
      imageData.data[px + 1] = normalized[i];
      imageData.data[px + 2] = normalized[i];
      imageData.data[px + 3] = 255;
    }
  }

  return imageData;
}

export function sliceToCanvas(
  slice: Float32Array,
  width: number,
  height: number,
  maskSlice?: Float32Array
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const imageData = sliceToImageData(slice, width, height, maskSlice);
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

export function canvasToPNG(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob!), 'image/png');
  });
}
