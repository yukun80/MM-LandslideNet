#!/usr/bin/env python3
"""
Linux Compatibility Fix Script
Apply all necessary fixes for Linux environment compatibility
"""

import os
import sys
import shutil
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

# Try to import Config, but handle missing dependencies gracefully
try:
    from configs.config import Config

    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import Config ({e})")
    print("   This is expected if PyTorch is not installed yet.")
    CONFIG_AVAILABLE = False


def set_executable_permissions():
    """Set executable permissions for Python scripts"""
    print("🔧 Setting executable permissions for Python scripts...")

    script_files = [
        "scripts/check_environment.py",
        "scripts/step1_data_visualization_demo.py",
        "scripts/step1_data_visualization_comprehensive.py",
        "scripts/step2_data_quality_assessment_01.py",
        "scripts/step2_analyze_quality_02.py",
        "scripts/step2_calculate_clean_stats_03.py",
        "optical_src/train.py",
        "optical_src/inference.py",
    ]

    for script_file in script_files:
        script_path = Path(script_file)
        if script_path.exists():
            # Add executable permission
            os.chmod(script_path, 0o755)
            print(f"   ✅ {script_file}")
        else:
            print(f"   ❌ {script_file} - File not found")


def ensure_shebang_lines():
    """Ensure all Python scripts have proper shebang lines"""
    print("\n🔧 Checking shebang lines...")

    scripts_to_check = [
        "scripts/check_environment.py",
        "scripts/step1_data_visualization_demo.py",
        "scripts/step1_data_visualization_comprehensive.py",
        "scripts/step2_data_quality_assessment_01.py",
        "scripts/step2_analyze_quality_02.py",
        "scripts/step2_calculate_clean_stats_03.py",
        "optical_src/train.py",
        "optical_src/inference.py",
    ]

    for script_file in scripts_to_check:
        script_path = Path(script_file)
        if script_path.exists():
            with open(script_path, "r") as f:
                first_line = f.readline().strip()

            if not first_line.startswith("#!"):
                print(f"   ⚠️  {script_file} - Missing shebang line")
                # Read the file content
                with open(script_path, "r") as f:
                    content = f.read()

                # Add shebang line
                with open(script_path, "w") as f:
                    f.write("#!/usr/bin/env python3\n" + content)

                print(f"   ✅ Added shebang to {script_file}")
            else:
                print(f"   ✅ {script_file} - Shebang OK")


def check_directory_structure():
    """Check and create necessary directories"""
    print("\n📁 Checking directory structure...")

    # Use hardcoded paths if Config is not available
    if CONFIG_AVAILABLE:
        config = Config()
        directories = [
            config.OUTPUT_ROOT,
            config.CHECKPOINT_DIR,
            config.SUBMISSION_DIR,
            config.LOG_DIR,
            config.DATASET_ROOT / "data_check",
        ]
    else:
        # Fallback to hardcoded paths
        project_root = Path(__file__).parent.parent
        directories = [
            project_root / "outputs",
            project_root / "outputs" / "checkpoints",
            project_root / "outputs" / "submissions",
            project_root / "logs",
            project_root / "dataset" / "data_check",
        ]

    for directory in directories:
        if directory.exists():
            print(f"   ✅ {directory}")
        else:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   ➕ Created {directory}")


def create_linux_run_scripts():
    """Create convenience scripts for Linux environment"""
    print("\n📝 Creating Linux convenience scripts...")

    # Create data quality assessment runner
    quality_runner = """#!/bin/bash
# Data Quality Assessment Pipeline Runner for Linux
echo "🚀 Running Data Quality Assessment Pipeline"
echo "============================================="

# Step 1: RGB-based quality assessment
echo "📸 Step 1: RGB-based quality assessment..."
python3 scripts/step2_data_quality_assessment_01.py

if [ $? -eq 0 ]; then
    echo "✅ Step 1 completed successfully"
else
    echo "❌ Step 1 failed"
    exit 1
fi

# Step 2: Quality analysis and threshold determination
echo "📊 Step 2: Quality analysis and threshold determination..."
python3 scripts/step2_analyze_quality_02.py

if [ $? -eq 0 ]; then
    echo "✅ Step 2 completed successfully"
else
    echo "❌ Step 2 failed"
    exit 1
fi

# Step 3: Clean dataset statistics calculation
echo "📈 Step 3: Clean dataset statistics calculation..."
python3 scripts/step2_calculate_clean_stats_03.py

if [ $? -eq 0 ]; then
    echo "✅ Step 3 completed successfully"
    echo "🎉 Data quality assessment pipeline completed!"
else
    echo "❌ Step 3 failed"
    exit 1
fi
"""

    with open("scripts/run_quality_assessment.sh", "w") as f:
        f.write(quality_runner)

    os.chmod("scripts/run_quality_assessment.sh", 0o755)
    print("   ✅ Created scripts/run_quality_assessment.sh")

    # Create optical training runner
    training_runner = """#!/bin/bash
# Optical Baseline Training Runner for Linux
echo "🚀 Running Optical Baseline Training"
echo "===================================="

# Check environment first
echo "🔍 Checking environment..."
python3 scripts/check_environment.py

if [ $? -eq 0 ]; then
    echo "✅ Environment check passed"
else
    echo "❌ Environment check failed"
    exit 1
fi

# Run training
echo "🎓 Starting optical baseline training..."
python3 optical_src/train.py

if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully"
else
    echo "❌ Training failed"
    exit 1
fi
"""

    with open("scripts/run_optical_training.sh", "w") as f:
        f.write(training_runner)

    os.chmod("scripts/run_optical_training.sh", 0o755)
    print("   ✅ Created scripts/run_optical_training.sh")

    # Create inference runner
    inference_runner = """#!/bin/bash
# Optical Baseline Inference Runner for Linux
echo "🚀 Running Optical Baseline Inference"
echo "====================================="

# Check environment first
echo "🔍 Checking environment..."
python3 scripts/check_environment.py

if [ $? -eq 0 ]; then
    echo "✅ Environment check passed"
else
    echo "❌ Environment check failed"
    exit 1
fi

# Run inference
echo "🔮 Starting optical baseline inference..."
python3 optical_src/inference.py

if [ $? -eq 0 ]; then
    echo "✅ Inference completed successfully"
else
    echo "❌ Inference failed"
    exit 1
fi
"""

    with open("scripts/run_optical_inference.sh", "w") as f:
        f.write(inference_runner)

    os.chmod("scripts/run_optical_inference.sh", 0o755)
    print("   ✅ Created scripts/run_optical_inference.sh")


def create_requirements_txt():
    """Create requirements.txt for easy dependency installation"""
    print("\n📦 Creating requirements.txt...")

    requirements = """# MM-LandslideNet Requirements
# Core ML frameworks
torch>=1.12.0
torchvision>=0.13.0
timm>=0.6.0

# Data processing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0

# Image processing
opencv-python>=4.5.0
albumentations>=1.2.0
Pillow>=8.3.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Progress bars and utilities
tqdm>=4.60.0
pathlib2>=2.3.0

# Logging and monitoring
tensorboard>=2.8.0
wandb>=0.12.0

# Jupyter notebook support
jupyter>=1.0.0
notebook>=6.4.0
ipywidgets>=7.6.0

# Additional utilities
psutil>=5.8.0
pyyaml>=6.0.0
"""

    with open("requirements.txt", "w") as f:
        f.write(requirements)

    print("   ✅ Created requirements.txt")


def create_setup_script():
    """Create setup script for Linux environment"""
    print("\n⚙️  Creating setup script...")

    setup_script = """#!/bin/bash
# MM-LandslideNet Setup Script for Linux
echo "🚀 Setting up MM-LandslideNet for Linux"
echo "======================================="

# Check Python version
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.7 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Run environment check
echo "🔍 Running environment check..."
python3 scripts/check_environment.py

echo "✅ Setup completed successfully!"
echo "📝 To activate the environment, run: source venv/bin/activate"
"""

    with open("setup_linux.sh", "w") as f:
        f.write(setup_script)

    os.chmod("setup_linux.sh", 0o755)
    print("   ✅ Created setup_linux.sh")


def main():
    """Main function"""
    print("🔧 Linux Compatibility Fix Script")
    print("=" * 40)

    set_executable_permissions()
    ensure_shebang_lines()
    check_directory_structure()
    create_linux_run_scripts()
    create_requirements_txt()
    create_setup_script()

    print("\n✅ All Linux compatibility fixes applied!")
    print("=" * 40)
    print("📝 Next steps:")
    print("   1. Run: ./setup_linux.sh")
    print("   2. Activate environment: source venv/bin/activate")
    print("   3. Check environment: python3 scripts/check_environment.py")
    print("   4. Run quality assessment: ./scripts/run_quality_assessment.sh")
    print("   5. Run training: ./scripts/run_optical_training.sh")


if __name__ == "__main__":
    main()
