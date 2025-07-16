# MM-LandslideNet Linux Environment Setup Guide

## 🚀 Quick Start

This guide helps you set up the MM-LandslideNet project in a Linux environment. The project has been adapted from Windows to ensure full compatibility with Linux systems.

### 1. Prerequisites

- Python 3.7 or higher
- pip package manager
- Git (optional, for version control)

### 2. Setup Steps

#### Step 1: Run the Setup Script
```bash
chmod +x setup_linux.sh
./setup_linux.sh
```

This script will:
- Create a virtual environment
- Install all required dependencies
- Run an environment check

#### Step 2: Activate the Environment
```bash
source venv/bin/activate
```

#### Step 3: Verify Installation
```bash
python3 scripts/check_environment.py
```

## 🔧 Linux Compatibility Fixes Applied

### 1. Configuration Changes

**File: `configs/config.py`**
- ✅ Fixed CUDA device detection to use `torch.cuda.is_available()` instead of hardcoded paths
- ✅ Removed Windows-specific CUDA path checking

### 2. Script Permissions

All Python scripts now have:
- ✅ Proper shebang lines (`#!/usr/bin/env python3`)
- ✅ Executable permissions (`chmod +x`)

### 3. Path Handling

All file paths use:
- ✅ `pathlib.Path` for cross-platform compatibility
- ✅ Forward slashes (`/`) for path separation
- ✅ Relative paths instead of absolute paths

### 4. New Convenience Scripts

**Quality Assessment Pipeline:**
```bash
./scripts/run_quality_assessment.sh
```

**Optical Baseline Training:**
```bash
./scripts/run_optical_training.sh
```

**Optical Baseline Inference:**
```bash
./scripts/run_optical_inference.sh
```

## 📂 Project Structure

```
MM-LandslideNet_202507/
├── configs/                    # Configuration files
│   └── config.py              # Main configuration (Linux-compatible)
├── optical_src/               # Optical baseline model
│   ├── train.py              # Training script
│   ├── inference.py          # Inference script
│   ├── model.py              # Model architecture
│   ├── dataset.py            # Dataset handling
│   └── utils.py              # Utility functions
├── scripts/                   # Data processing scripts
│   ├── check_environment.py   # Environment validation
│   ├── step2_data_quality_assessment_01.py
│   ├── step2_analyze_quality_02.py
│   ├── step2_calculate_clean_stats_03.py
│   ├── run_quality_assessment.sh
│   ├── run_optical_training.sh
│   └── run_optical_inference.sh
├── dataset/                   # Data directory
│   ├── Train.csv
│   ├── Test.csv
│   ├── train_data/           # Training .npy files
│   ├── test_data/            # Test .npy files
│   └── data_check/           # Quality assessment results
├── outputs/                   # Generated outputs
│   ├── checkpoints/          # Model checkpoints
│   ├── submissions/          # Competition submissions
│   └── logs/                 # Log files
├── requirements.txt          # Python dependencies
├── setup_linux.sh           # Linux setup script
└── LINUX_SETUP_README.md    # This file
```

## 🎯 Usage Examples

### 1. Data Quality Assessment

Run the complete data quality assessment pipeline:
```bash
./scripts/run_quality_assessment.sh
```

Or run individual steps:
```bash
# Step 1: RGB-based quality assessment
python3 scripts/step2_data_quality_assessment_01.py

# Step 2: Quality analysis and threshold determination
python3 scripts/step2_analyze_quality_02.py

# Step 3: Clean dataset statistics calculation
python3 scripts/step2_calculate_clean_stats_03.py
```

### 2. Optical Baseline Training

```bash
./scripts/run_optical_training.sh
```

Or run directly:
```bash
python3 optical_src/train.py
```

### 3. Optical Baseline Inference

```bash
./scripts/run_optical_inference.sh
```

Or run directly:
```bash
python3 optical_src/inference.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Permission Denied Error**
   ```bash
   chmod +x scripts/*.py
   chmod +x scripts/*.sh
   ```

2. **Module Not Found Error**
   ```bash
   # Make sure virtual environment is activated
   source venv/bin/activate
   
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

3. **CUDA Not Available**
   - The system will automatically fall back to CPU
   - Check with: `python3 -c "import torch; print(torch.cuda.is_available())"`

4. **Path Issues**
   - All paths now use `pathlib.Path` for cross-platform compatibility
   - No manual path fixing should be needed

### Environment Check

Always run the environment check first:
```bash
python3 scripts/check_environment.py
```

This will verify:
- ✅ System information
- ✅ PyTorch and CUDA availability
- ✅ Project directory structure
- ✅ Data file existence
- ✅ Configuration validity
- ✅ Dependency installation

## 🔄 Migration from Windows

If you're migrating from Windows, the following changes have been applied automatically:

1. **Device Detection**: CUDA detection now uses proper PyTorch APIs
2. **File Permissions**: All scripts are now executable
3. **Path Separators**: All paths use forward slashes
4. **Shebang Lines**: All Python scripts have proper shebang lines
5. **Shell Scripts**: Added convenient bash scripts for common tasks

## 📦 Dependencies

The project now includes a complete `requirements.txt` with all necessary dependencies:

- **Core ML**: torch, torchvision, timm
- **Data Processing**: numpy, pandas, scikit-learn
- **Image Processing**: opencv-python, albumentations, Pillow
- **Visualization**: matplotlib, seaborn, plotly
- **Utilities**: tqdm, tensorboard, jupyter

## 🎉 Success Verification

After setup, you should be able to:
- ✅ Run `python3 scripts/check_environment.py` without errors
- ✅ See proper CUDA detection (or CPU fallback)
- ✅ Execute all scripts with proper permissions
- ✅ Access all project directories
- ✅ Load and process data files

---

**Note**: This setup has been tested on Ubuntu 20.04 with Python 3.8+. For other Linux distributions, minor adjustments may be needed. 