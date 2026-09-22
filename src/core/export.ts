import * as XLSX from 'xlsx';
import type { FeatureResult } from './features';

export function exportToXLSX(results: FeatureResult[], fileName: string = 'results.xlsx'): void {
  const ws = XLSX.utils.json_to_sheet(results);
  const colWidths = Object.keys(results[0] || {}).map((key) => {
    const maxLen = Math.max(
      key.length,
      ...results.map((r) => String(r[key] ?? '').length)
    );
    return { wch: maxLen + 2 };
  });
  ws['!cols'] = colWidths;

  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, 'Features');

  XLSX.writeFile(wb, fileName);
}

export function downloadBlob(blob: Blob, fileName: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
