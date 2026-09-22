import { useState, useCallback, useRef } from 'react';
import {
  Upload, FileImage, FileCheck2, Loader2, Download, Table, AlertCircle,
  X, Brain, Activity, Layers, ChevronRight, Info, CheckCircle2,
} from 'lucide-react';
import { parseNRRD, type NRRDData, getSlice, normalizeSlice } from '@/core/nrrd';
import { extractAllSlices, type FeatureResult } from '@/core/features';
import { exportToXLSX, downloadBlob } from '@/core/export';

type Status = 'idle' | 'loading' | 'processing' | 'done' | 'error';

interface FileSlot {
  file: File | null;
  nrrd: NRRDData | null;
  preview: string | null;
}

export default function App() {
  const [imageSlot, setImageSlot] = useState<FileSlot>({ file: null, nrrd: null, preview: null });
  const [maskSlot, setMaskSlot] = useState<FileSlot>({ file: null, nrrd: null, preview: null });
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string>('');
  const [results, setResults] = useState<FeatureResult[]>([]);
  const [currentSlice, setCurrentSlice] = useState(0);
  const [maxSlice, setMaxSlice] = useState(0);
  const [showResults, setShowResults] = useState(false);
  const [dragOver, setDragOver] = useState<'image' | 'mask' | null>(null);
  const [processedImagePreview, setProcessedImagePreview] = useState<string | null>(null);
  const [processedMaskPreview, setProcessedMaskPreview] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleFile = useCallback(async (file: File, type: 'image' | 'mask') => {
    const setSlot = type === 'image' ? setImageSlot : setMaskSlot;
    setStatus('loading');
    setError('');

    try {
      const isNrrd = file.name.endsWith('.nrrd') && !file.name.endsWith('.seg.nrrd');
      const isSegNrrd = file.name.endsWith('.seg.nrrd') || file.name.endsWith('.seg.nrrd');

      if (type === 'image' && !isNrrd) {
        throw new Error('Image file must be a .nrrd file');
      }
      if (type === 'mask' && !file.name.endsWith('.nrrd')) {
        throw new Error('Mask file must be a .seg.nrrd file');
      }

      const nrrd = await parseNRRD(file);
      const shape = nrrd.shape;
      const depth = shape.length >= 3 ? shape[0] : 1;
      const width = shape.length >= 3 ? shape[2] : (shape.length >= 2 ? shape[1] : shape[0]);
      const height = shape.length >= 3 ? shape[1] : (shape.length >= 2 ? shape[0] : 1);

      const midSlice = Math.floor(depth / 2);
      const slice = getSlice(nrrd, midSlice, 0);
      const normalized = normalizeSlice(slice);

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d')!;
      const imageData = ctx.createImageData(width, height);
      for (let i = 0; i < normalized.length; i++) {
        const px = i * 4;
        imageData.data[px] = normalized[i];
        imageData.data[px + 1] = normalized[i];
        imageData.data[px + 2] = normalized[i];
        imageData.data[px + 3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);
      const preview = canvas.toDataURL('image/png');

      setSlot({ file, nrrd, preview });
      setStatus('idle');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to parse file');
      setStatus('error');
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent, type: 'image' | 'mask') => {
    e.preventDefault();
    setDragOver(null);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file, type);
  }, [handleFile]);

  const handleSliceChange = useCallback((sliceIdx: number) => {
    if (!imageSlot.nrrd || !maskSlot.nrrd) return;
    setCurrentSlice(sliceIdx);

    const shape = imageSlot.nrrd.shape;
    const depth = shape.length >= 3 ? shape[0] : 1;
    const width = shape.length >= 3 ? shape[2] : (shape.length >= 2 ? shape[1] : shape[0]);
    const height = shape.length >= 3 ? shape[1] : (shape.length >= 2 ? shape[0] : 1);

    const imgSlice = getSlice(imageSlot.nrrd, sliceIdx, 0);
    const maskSlice = getSlice(maskSlot.nrrd, sliceIdx, 0);
    const imgNorm = normalizeSlice(imgSlice);
    const maskNorm = normalizeSlice(maskSlice);

    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.createImageData(width, height);

    for (let i = 0; i < imgNorm.length; i++) {
      const px = i * 4;
      if (maskNorm[i] > 0) {
        imageData.data[px] = Math.round(imgNorm[i] * 0.4 + 200);
        imageData.data[px + 1] = Math.round(imgNorm[i] * 0.4 + 50);
        imageData.data[px + 2] = Math.round(imgNorm[i] * 0.4 + 50);
      } else {
        imageData.data[px] = imgNorm[i];
        imageData.data[px + 1] = imgNorm[i];
        imageData.data[px + 2] = imgNorm[i];
      }
      imageData.data[px + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);

    setProcessedImagePreview(canvas.toDataURL('image/png'));
  }, [imageSlot.nrrd, maskSlot.nrrd]);

  const process = useCallback(async () => {
    if (!imageSlot.nrrd || !maskSlot.nrrd) return;
    setStatus('processing');
    setError('');
    setShowResults(false);

    try {
      await new Promise(r => setTimeout(r, 100));

      const depth = Math.min(
        imageSlot.nrrd.shape.length >= 3 ? imageSlot.nrrd.shape[0] : 1,
        50
      );
      setMaxSlice(depth);

      const allResults = extractAllSlices(imageSlot.nrrd, maskSlot.nrrd, depth);
      setResults(allResults);

      const midSlice = Math.floor(depth / 2);
      setCurrentSlice(midSlice);
      handleSliceChange(midSlice);

      const shape = imageSlot.nrrd.shape;
      const width = shape.length >= 3 ? shape[2] : (shape.length >= 2 ? shape[1] : shape[0]);
      const height = shape.length >= 3 ? shape[1] : (shape.length >= 2 ? shape[0] : 1);

      const imgSlice = getSlice(imageSlot.nrrd, midSlice, 0);
      const maskSlice = getSlice(maskSlot.nrrd, midSlice, 0);
      const imgNorm = normalizeSlice(imgSlice);
      const maskNorm = normalizeSlice(maskSlice);

      const imgCanvas = document.createElement('canvas');
      imgCanvas.width = width;
      imgCanvas.height = height;
      const imgCtx = imgCanvas.getContext('2d')!;
      const imgData = imgCtx.createImageData(width, height);
      for (let i = 0; i < imgNorm.length; i++) {
        const px = i * 4;
        imgData.data[px] = imgNorm[i];
        imgData.data[px + 1] = imgNorm[i];
        imgData.data[px + 2] = imgNorm[i];
        imgData.data[px + 3] = 255;
      }
      imgCtx.putImageData(imgData, 0, 0);
      setProcessedImagePreview(imgCanvas.toDataURL('image/png'));

      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = width;
      maskCanvas.height = height;
      const maskCtx = maskCanvas.getContext('2d')!;
      const maskData = maskCtx.createImageData(width, height);
      for (let i = 0; i < maskNorm.length; i++) {
        const px = i * 4;
        if (maskNorm[i] > 0) {
          maskData.data[px] = 220;
          maskData.data[px + 1] = 60;
          maskData.data[px + 2] = 60;
        } else {
          maskData.data[px] = 20;
          maskData.data[px + 1] = 20;
          maskData.data[px + 2] = 20;
        }
        maskData.data[px + 3] = 255;
      }
      maskCtx.putImageData(maskData, 0, 0);
      setProcessedMaskPreview(maskCanvas.toDataURL('image/png'));

      setStatus('done');
      setShowResults(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Processing failed');
      setStatus('error');
    }
  }, [imageSlot.nrrd, maskSlot.nrrd, handleSliceChange]);

  const downloadXLSX = useCallback(() => {
    if (results.length === 0) return;
    exportToXLSX(results, 'radiomics_results.xlsx');
  }, [results]);

  const downloadProcessedImage = useCallback(async () => {
    if (!processedImagePreview) return;
    const res = await fetch(processedImagePreview);
    const blob = await res.blob();
    downloadBlob(blob, 'processed_image.png');
  }, [processedImagePreview]);

  const downloadProcessedMask = useCallback(async () => {
    if (!processedMaskPreview) return;
    const res = await fetch(processedMaskPreview);
    const blob = await res.blob();
    downloadBlob(blob, 'processed_mask.png');
  }, [processedMaskPreview]);

  const reset = useCallback(() => {
    setImageSlot({ file: null, nrrd: null, preview: null });
    setMaskSlot({ file: null, nrrd: null, preview: null });
    setStatus('idle');
    setError('');
    setResults([]);
    setShowResults(false);
    setProcessedImagePreview(null);
    setProcessedMaskPreview(null);
  }, []);

  const canProcess = imageSlot.nrrd && maskSlot.nrrd && status !== 'processing';

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-700/50 backdrop-blur-md bg-slate-900/70 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">NeuroRad Analytics</h1>
              <p className="text-xs text-slate-400">NRRD Radiomics Feature Extractor</p>
            </div>
          </div>
          <div className="flex items-center gap-4 text-sm text-slate-400">
            <span className="hidden sm:flex items-center gap-1.5">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>Web Edition</span>
            </span>
            <button
              onClick={reset}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 transition-colors text-sm"
            >
              Reset
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Upload Section */}
        {!showResults && (
          <div className="space-y-8">
            <div className="text-center max-w-2xl mx-auto">
              <h2 className="text-3xl font-bold mb-3 bg-gradient-to-r from-cyan-300 to-blue-400 bg-clip-text text-transparent">
                Medical Image Radiomics Analysis
              </h2>
              <p className="text-slate-400">
                Upload a NRRD image and its segmentation mask to extract quantitative radiomic features.
                Results are exported as an Excel spreadsheet with processed image previews.
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              {/* Image Upload */}
              <UploadCard
                title="Image File"
                subtitle=".nrrd format"
                icon={<FileImage className="w-8 h-8" />}
                accent="cyan"
                slot={imageSlot}
                dragOver={dragOver === 'image'}
                onDrop={(e) => handleDrop(e, 'image')}
                onDragOver={(e) => { e.preventDefault(); setDragOver('image'); }}
                onDragLeave={() => setDragOver(null)}
                onFileSelect={(f) => handleFile(f, 'image')}
              />

              {/* Mask Upload */}
              <UploadCard
                title="Segmentation Mask"
                subtitle=".seg.nrrd format"
                icon={<Layers className="w-8 h-8" />}
                accent="blue"
                slot={maskSlot}
                dragOver={dragOver === 'mask'}
                onDrop={(e) => handleDrop(e, 'mask')}
                onDragOver={(e) => { e.preventDefault(); setDragOver('mask'); }}
                onDragLeave={() => setDragOver(null)}
                onFileSelect={(f) => handleFile(f, 'mask')}
              />
            </div>

            {/* Error */}
            {error && (
              <div className="max-w-2xl mx-auto flex items-center gap-3 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <span className="text-sm">{error}</span>
                <button onClick={() => setError('')} className="ml-auto text-red-300 hover:text-red-200">
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* Process Button */}
            <div className="flex justify-center">
              <button
                onClick={process}
                disabled={!canProcess}
                className={`px-8 py-3.5 rounded-xl font-semibold text-base transition-all flex items-center gap-2.5 ${
                  canProcess
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 shadow-lg shadow-cyan-500/30 hover:shadow-cyan-500/50 hover:scale-105'
                    : 'bg-slate-700 text-slate-500 cursor-not-allowed'
                }`}
              >
                {status === 'processing' ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <ChevronRight className="w-5 h-5" />
                    Extract Features
                  </>
                )}
              </button>
            </div>

            {/* Info */}
            <div className="max-w-2xl mx-auto flex items-start gap-3 px-4 py-3 rounded-xl bg-slate-800/50 border border-slate-700/50 text-slate-300">
              <Info className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
              <div className="text-sm space-y-1">
                <p>The analysis extracts 24 radiomic features per slice, including:</p>
                <p className="text-slate-400">First-order statistics (mean, std, skewness, kurtosis, entropy), GLCM texture features (contrast, correlation, energy, homogeneity), and morphological features (area, perimeter, eccentricity, extent).</p>
              </div>
            </div>
          </div>
        )}

        {/* Results Section */}
        {showResults && (
          <div className="space-y-6 animate-in fade-in duration-500">
            {/* Success Banner */}
            <div className="flex items-center gap-3 px-5 py-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30">
              <CheckCircle2 className="w-6 h-6 text-emerald-400 flex-shrink-0" />
              <div>
                <p className="font-semibold text-emerald-300">Analysis Complete</p>
                <p className="text-sm text-emerald-400/70">
                  Extracted features from {results.length} slices. Download results and processed images below.
                </p>
              </div>
            </div>

            {/* Processed Images */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-slate-800/50 rounded-2xl border border-slate-700/50 overflow-hidden">
                <div className="px-5 py-3 border-b border-slate-700/50 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <FileImage className="w-5 h-5 text-cyan-400" />
                    <h3 className="font-semibold text-sm">Processed Image</h3>
                  </div>
                  <button
                    onClick={downloadProcessedImage}
                    className="text-xs px-3 py-1.5 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 transition-colors flex items-center gap-1.5"
                  >
                    <Download className="w-3.5 h-3.5" />
                    PNG
                  </button>
                </div>
                <div className="p-4 flex items-center justify-center bg-slate-900/50 min-h-[300px]">
                  {processedImagePreview && (
                    <img src={processedImagePreview} alt="Processed" className="max-w-full max-h-[400px] rounded-lg" />
                  )}
                </div>
              </div>

              <div className="bg-slate-800/50 rounded-2xl border border-slate-700/50 overflow-hidden">
                <div className="px-5 py-3 border-b border-slate-700/50 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Layers className="w-5 h-5 text-blue-400" />
                    <h3 className="font-semibold text-sm">Processed Mask</h3>
                  </div>
                  <button
                    onClick={downloadProcessedMask}
                    className="text-xs px-3 py-1.5 rounded-lg bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 transition-colors flex items-center gap-1.5"
                  >
                    <Download className="w-3.5 h-3.5" />
                    PNG
                  </button>
                </div>
                <div className="p-4 flex items-center justify-center bg-slate-900/50 min-h-[300px]">
                  {processedMaskPreview && (
                    <img src={processedMaskPreview} alt="Mask" className="max-w-full max-h-[400px] rounded-lg" />
                  )}
                </div>
              </div>
            </div>

            {/* Slice Navigator */}
            {maxSlice > 1 && (
              <div className="bg-slate-800/50 rounded-2xl border border-slate-700/50 p-5">
                <div className="flex items-center gap-3 mb-3">
                  <Layers className="w-5 h-5 text-cyan-400" />
                  <h3 className="font-semibold text-sm">Slice Navigator</h3>
                  <span className="text-xs text-slate-400 ml-auto">
                    Slice {currentSlice + 1} / {maxSlice}
                  </span>
                </div>
                <div className="flex items-center gap-4">
                  <input
                    type="range"
                    min={0}
                    max={maxSlice - 1}
                    value={currentSlice}
                    onChange={(e) => handleSliceChange(parseInt(e.target.value))}
                    className="flex-1 accent-cyan-500"
                  />
                  <canvas ref={canvasRef} className="hidden" />
                </div>
                {processedImagePreview && (
                  <div className="mt-4 flex justify-center">
                    <img src={processedImagePreview} alt="Slice" className="max-h-[200px] rounded-lg border border-slate-700" />
                  </div>
                )}
              </div>
            )}

            {/* Results Table */}
            <div className="bg-slate-800/50 rounded-2xl border border-slate-700/50 overflow-hidden">
              <div className="px-5 py-3 border-b border-slate-700/50 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Table className="w-5 h-5 text-cyan-400" />
                  <h3 className="font-semibold text-sm">Feature Results ({results.length} slices)</h3>
                </div>
                <button
                  onClick={downloadXLSX}
                  className="text-sm px-4 py-2 rounded-lg bg-gradient-to-r from-emerald-500 to-green-600 hover:from-emerald-400 hover:to-green-500 transition-all flex items-center gap-2 font-medium shadow-lg shadow-emerald-500/20"
                >
                  <Download className="w-4 h-4" />
                  Download Excel
                </button>
              </div>
              <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="bg-slate-900/50 sticky top-0">
                    <tr>
                      {results.length > 0 && Object.keys(results[0]).map((key) => (
                        <th key={key} className="px-3 py-2.5 text-left font-semibold text-slate-300 whitespace-nowrap border-b border-slate-700/50">
                          {key}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((row, i) => (
                      <tr key={i} className="hover:bg-slate-700/30 transition-colors border-b border-slate-700/20">
                        {Object.values(row).map((val, j) => (
                          <td key={j} className="px-3 py-2 text-slate-300 whitespace-nowrap">
                            {typeof val === 'number' ? val.toFixed(4) : val}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* New Analysis Button */}
            <div className="flex justify-center pt-2">
              <button
                onClick={reset}
                className="px-6 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 transition-colors flex items-center gap-2"
              >
                <Upload className="w-4 h-4" />
                New Analysis
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-700/50 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-4 text-center text-sm text-slate-500">
          NeuroRad Analytics - NRRD Radiomics Feature Extractor
        </div>
      </footer>
    </div>
  );
}

interface UploadCardProps {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  accent: 'cyan' | 'blue';
  slot: FileSlot;
  dragOver: boolean;
  onDrop: (e: React.DragEvent) => void;
  onDragOver: (e: React.DragEvent) => void;
  onDragLeave: () => void;
  onFileSelect: (file: File) => void;
}

function UploadCard({ title, subtitle, icon, accent, slot, dragOver, onDrop, onDragOver, onDragLeave, onFileSelect }: UploadCardProps) {
  const accentClasses = {
    cyan: 'border-cyan-500/50 bg-cyan-500/5',
    blue: 'border-blue-500/50 bg-blue-500/5',
  };
  const iconClasses = {
    cyan: 'text-cyan-400 bg-cyan-500/10',
    blue: 'text-blue-400 bg-blue-500/10',
  };

  return (
    <div
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      className={`relative rounded-2xl border-2 border-dashed transition-all duration-300 ${
        dragOver ? accentClasses[accent] + ' scale-105' : 'border-slate-600/50 hover:border-slate-500'
      } ${slot.file ? 'bg-slate-800/30' : 'bg-slate-800/20'} p-6`}
    >
      {!slot.file ? (
        <label className="cursor-pointer flex flex-col items-center justify-center text-center min-h-[200px]">
          <div className={`w-16 h-16 rounded-2xl ${iconClasses[accent]} flex items-center justify-center mb-4 transition-transform group-hover:scale-110`}>
            {icon}
          </div>
          <h3 className="font-semibold text-base mb-1">{title}</h3>
          <p className="text-sm text-slate-400 mb-4">{subtitle}</p>
          <div className="flex items-center gap-2 text-sm text-slate-300 bg-slate-700/50 px-4 py-2 rounded-lg hover:bg-slate-700 transition-colors">
            <Upload className="w-4 h-4" />
            Choose File
          </div>
          <input
            type="file"
            accept=".nrrd"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && onFileSelect(e.target.files[0])}
          />
        </label>
      ) : (
        <div className="flex flex-col items-center text-center min-h-[200px]">
          <div className={`w-16 h-16 rounded-2xl ${iconClasses[accent]} flex items-center justify-center mb-4`}>
            <FileCheck2 className="w-8 h-8" />
          </div>
          <h3 className="font-semibold text-base mb-1">{title}</h3>
          <p className="text-sm text-slate-400 truncate max-w-full mb-3">{slot.file.name}</p>
          {slot.preview && (
            <img src={slot.preview} alt="Preview" className="max-h-[120px] rounded-lg border border-slate-700 mb-3" />
          )}
          {slot.nrrd && (
            <p className="text-xs text-slate-500">
              Shape: [{slot.nrrd.shape.join(', ')}]
            </p>
          )}
          <label className="cursor-pointer text-sm text-cyan-400 hover:text-cyan-300 mt-2">
            Change File
            <input
              type="file"
              accept=".nrrd"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && onFileSelect(e.target.files[0])}
            />
          </label>
        </div>
      )}
    </div>
  );
}
