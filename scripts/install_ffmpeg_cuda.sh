#!/bin/bash

set -euo pipefail
set -x

# Step 1: Install CUDA Toolkit
# Add the NVIDIA repository for CUDA and install CUDA toolkit version 12.6
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf clean all
dnf -y install cuda-toolkit-12-6
# Set CUDA environment variables for PATH and LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
# Verify nvcc installation (CUDA compiler)
nvcc --version
# Step 2: Install dependencies required for FFmpeg
# Install libraries and tools required for FFmpeg compilation
dnf install -y libtool \
    glibc \
    glibc-devel \
    numactl \
    numactl-devel \
    openssl \
    yasm \
    pkg-config \
    openssl-devel \
    git \
    gcc \
    make \
    gcc-c++ \
    kernel-headers \
    automake
# Step 3: Install x264
# Add the RPM Fusion repository and install x264 and its development files
dnf install -y https://download1.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm
dnf install -y x264 x264-devel
# Step 4: Install NVENC codec headers
# Install the necessary development tools and clone the nv-codec-headers repository
# Clone the nv-codec-headers repository and install it
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
git checkout 9934f17316b66ce6de12f3b82203a298bc9351d8 # Fix the version
make
make install
cd ..
# Set PKG_CONFIG_PATH for pkg-config to find the newly installed nv-codec-headers
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH:-}

# Add /usr/local/lib to the library search path and update the dynamic linker cache
echo "/usr/local/lib" | tee -a /etc/ld.so.conf
ldconfig
# Step 5: Install FFmpeg
# Clone the FFmpeg repository
git clone https://git.ffmpeg.org/ffmpeg.git
# Configure and compile FFmpeg with necessary flags for NVIDIA, x264, and other libraries
cd ffmpeg
git checkout faa366003b58ba26484070ca408be4b9d5473a73 # Fix the version
./configure --enable-nonfree \
    --enable-cuda-nvcc \
    --enable-libnpp \
    --enable-libx264 \
    --enable-openssl \
    --enable-nvenc \
    --enable-gpl \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --disable-static \
    --enable-shared
# Compile and install FFmpeg
make -j $(nproc)
make install
# Do some cleanup
rm -rf /nv-codec-headers
rm -rf /ffmpeg
dnf remove -y \
    make \
    git \
    automake
dnf clean all
set +x

