#!/bin/bash
# Build and run Orbbec Femto Bolt Calibration Extractor

set -e  # Exit on error

echo "=========================================="
echo "Orbbec Calibration Extractor Build Script"
echo "=========================================="

# Check if OrbbecSDK is installed
if ! ldconfig -p | grep -q OrbbecSDK; then
    echo "❌ OrbbecSDK not found!"
    echo ""
    echo "Please install OrbbecSDK first:"
    echo "  1. Download from: https://github.com/orbbec/OrbbecSDK"
    echo "  2. Extract and install:"
    echo "     cd OrbbecSDK-*"
    echo "     sudo ./install.sh"
    echo ""
    exit 1
fi

echo "✅ OrbbecSDK found"

# Create build directory
if [ -d "build" ]; then
    echo "🗑️  Removing old build directory..."
    rm -rf build
fi

mkdir build
cd build

# Configure
echo ""
echo "🔧 Configuring CMake..."
cmake ..

# Build
echo ""
echo "🔨 Building..."
make -j$(nproc)

# Check if camera is connected
echo ""
echo "📡 Checking for Orbbec camera..."
if lsusb | grep -q "2bc5"; then
    echo "✅ Orbbec camera detected"
else
    echo "⚠️  No Orbbec camera detected"
    echo "   Please connect Femto Bolt and try again"
    exit 1
fi

# Run
echo ""
echo "=========================================="
echo "🚀 Running calibration extraction..."
echo "=========================================="
echo ""

./extract_calib

# Check results
echo ""
echo "=========================================="
echo "📂 Checking generated files..."
echo "=========================================="

cd ../../calib

if [ -f "rgb_camera_info.json" ]; then
    echo "✅ rgb_camera_info.json"
else
    echo "❌ rgb_camera_info.json NOT FOUND"
fi

if [ -f "depth_camera_info.json" ]; then
    echo "✅ depth_camera_info.json"
else
    echo "❌ depth_camera_info.json NOT FOUND"
fi

if [ -f "extrinsic_depth_to_color.json" ]; then
    echo "✅ extrinsic_depth_to_color.json"
else
    echo "❌ extrinsic_depth_to_color.json NOT FOUND"
fi

if [ -f "femto_bolt_calibration.txt" ]; then
    echo "✅ femto_bolt_calibration.txt"
else
    echo "❌ femto_bolt_calibration.txt NOT FOUND"
fi

if [ -f "femto_bolt_calibration_numpy.txt" ]; then
    echo "✅ femto_bolt_calibration_numpy.txt"
else
    echo "❌ femto_bolt_calibration_numpy.txt NOT FOUND"
fi

echo ""
echo "=========================================="
echo "✨ DONE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check calib/ directory for generated files"
echo "2. Review depth distortion coefficients in depth_camera_info.json"
echo "3. If distortion is near-zero, keep use_simple_resize=true"
echo "4. If distortion is significant, set use_simple_resize=false"
echo ""
