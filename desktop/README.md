# NeuroRad Analytics - Desktop Edition

## Building the Windows .exe

The desktop app is a standalone Python application that builds into a single
`.exe` file with no installation required on Windows 10.

### Prerequisites

1. Install Python 3.10+ from https://python.org
   - Check "Add Python to PATH" during installation

2. Install dependencies:
   ```cmd
   cd desktop
   pip install -r requirements.txt
   ```

### Build the .exe

```cmd
cd desktop
pyinstaller NeuroRad.spec
```

The standalone executable will be created at:
```
desktop/dist/NeuroRad.exe
```

Copy `NeuroRad.exe` to any Windows 10 machine and run it directly — no
installation needed.

### Alternative: one-command build

If you don't want to use the .spec file:

```cmd
pyinstaller --onefile --windowed --name NeuroRad desktop_app.py
```

### Running from source (without building exe)

```cmd
cd desktop
pip install -r requirements.txt
python desktop_app.py
```

### What the app does

1. User selects a `.nrrd` image file and a `.seg.nrrd` segmentation mask
2. The app processes each slice, extracting 24 radiomic features:
   - First-order statistics: mean, std, variance, skewness, kurtosis
   - Percentiles: min, max, range, median, P10, P25, P75, P90
   - Entropy
   - GLCM texture: contrast, correlation, energy, homogeneity
   - Morphology: area, perimeter, eccentricity, extent
3. Outputs three files:
   - `radiomics_results.xlsx` — Excel spreadsheet with all features per slice
   - `processed_image.png` — the middle slice with mask overlay
   - `processed_mask.png` — the middle slice of the segmentation mask
