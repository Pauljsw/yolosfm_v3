# Dense SFM Setup Guide

This guide explains how to enable Dense SFM reconstruction in COLMAP.

## ⚠️ Current Status

**Error when `dense: true`:**
```
ERROR: Dense stereo reconstruction requires CUDA, which is not available on your system.
```

## 🎯 Do You Need Dense?

**Short answer: NO for crack measurement!**

| Feature | Sparse | Dense |
|---------|--------|-------|
| **Point Count** | 10K-100K | 1M-10M+ |
| **Coverage** | Feature points only | Full surface |
| **Speed** | Fast (minutes) | Slow (hours) |
| **CUDA Required** | ❌ No | ✅ **YES** |
| **Crack Measurement** | ✅ Sufficient | ⚠️ Overkill |
| **Visualization** | Good | Better |

**Recommendation:** Use **Sparse** mode. It's sufficient for:
- Camera pose estimation
- Point cloud visualization with mask overlay
- Crack detection and measurement

Dense is only needed for:
- Photorealistic 3D reconstruction
- Texture mapping
- Dense meshing

---

## 🚀 How to Enable Dense (If You Really Want It)

### Option 1: Install CUDA on Your Current Machine

#### Step 1: Check GPU

```bash
# Check if you have NVIDIA GPU
lspci | grep -i nvidia

# Check CUDA availability
nvidia-smi
```

If `nvidia-smi` works, you already have CUDA drivers. Skip to Step 3.

#### Step 2: Install NVIDIA Drivers

**Ubuntu/Debian:**
```bash
# Add NVIDIA PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install latest driver
sudo apt install nvidia-driver-535  # Or latest version

# Reboot
sudo reboot

# Verify
nvidia-smi
```

#### Step 3: Install CUDA Toolkit

**Ubuntu 22.04:**
```bash
# Download CUDA installer
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run

# Install
sudo sh cuda_12.3.0_545.23.06_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

#### Step 4: Reinstall COLMAP with CUDA

**Option A: From source**
```bash
# Install dependencies
sudo apt install \
    git cmake build-essential \
    libboost-all-dev libeigen3-dev \
    libsuitesparse-dev libfreeimage-dev \
    libmetis-dev libgoogle-glog-dev \
    libgflags-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev \
    libcgal-dev libceres-dev

# Clone COLMAP
git clone https://github.com/colmap/colmap.git
cd colmap

# Build with CUDA
mkdir build
cd build
cmake .. -DCUDA_ENABLED=ON
make -j$(nproc)
sudo make install

# Verify
colmap --version
colmap stereo_fusion --help  # Should not error
```

**Option B: Docker (easier)**
```bash
# Pull COLMAP with CUDA
docker pull colmap/colmap:latest

# Run with GPU
docker run --gpus all -v $(pwd):/data colmap/colmap colmap --version
```

#### Step 5: Enable in Config

```yaml
# configs/simple.yaml
sfm:
  dense: true  # ← Enable
```

#### Step 6: Run

```bash
python -m src.pipeline sfm --config configs/simple.yaml
```

**Expected output:**
```
data/sfm/
├── sparse/0/          # Sparse reconstruction
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── dense/             # Dense reconstruction (NEW!)
    ├── fused.ply      # Dense point cloud (huge file!)
    ├── meshed-poisson.ply
    └── stereo/
```

---

### Option 2: Use Cloud GPU

If you don't have NVIDIA GPU locally:

**Google Colab (Free GPU):**
```python
# Install COLMAP in Colab
!apt install colmap

# Upload your images
# Run pipeline
```

**AWS EC2 (P2/P3 instances):**
```bash
# Launch P2.xlarge instance (NVIDIA K80)
# Pre-installed with CUDA
# Run your pipeline
```

---

### Option 3: Just Use Sparse (Recommended!)

**Current pipeline works perfectly with Sparse:**

1. ✅ SFM Sparse → Camera poses
2. ✅ Point Cloud Overlay → Red crack visualization
3. ✅ Pixel-to-MM → Accurate measurements
4. ✅ CSV Export → Results

**No Dense needed!**

---

## 📊 Dense vs Sparse Comparison

**Your dataset (197 images):**

| Mode | Time | Disk Space | CUDA |
|------|------|-----------|------|
| **Sparse** | ~5 min | ~50 MB | ❌ No |
| **Dense** | ~2 hours | ~5 GB | ✅ Required |

**Sparse Point Cloud:**
- 50,000 points
- Good for visualization
- Fast processing

**Dense Point Cloud:**
- 5,000,000+ points
- Beautiful but slow
- Needs powerful GPU

---

## 🎯 Recommendation

**For crack measurement: USE SPARSE!**

```yaml
sfm:
  dense: false  # ← Keep this
```

**Benefits:**
- No CUDA hassle
- Fast processing
- Same measurement accuracy
- Point cloud visualization works great

**Only enable Dense if:**
- You have NVIDIA GPU + CUDA installed
- You want photorealistic 3D model
- You have time to wait (hours)
- You have disk space (GB)

---

## 🆘 Troubleshooting

**"CUDA not available"**
- Check: `nvidia-smi`
- Reinstall COLMAP with CUDA support
- Or just use Sparse!

**"Out of memory"**
- Dense needs 8GB+ GPU RAM
- Reduce image resolution
- Or use Sparse!

**"Too slow"**
- Dense takes hours on large datasets
- Use GPU with 100+ CUDA cores
- Or use Sparse!

---

## ✅ TL;DR

1. **Sparse is enough** for crack measurement
2. Dense needs NVIDIA GPU + CUDA
3. Installation is complex
4. **Recommended: Keep `dense: false`**

Happy measuring! 🎉
