# Define base image.
FROM determinedai/environments:cuda-11.8-base-gpu-mpi-0.26.4
# Set environment variables.
ENV DET_MASTER="https://determined.corp.deepmirror.com:443"
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA architectures, required by tiny-cuda-nn. referecne to https://docs.nerf.studio/en/latest/quickstart/installation.html#
ENV TCNN_CUDA_ARCHITECTURES=70,80,86
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"
ENV CUDA_DEVICE_ORDER="PCI_BUS_ID"


# Install required apt packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libcgal-dev \
    libeigen3-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsuitesparse-dev \
    nano \
    protobuf-compiler \
    python3.8-dev \
    python3-pip \
    qtbase5-dev \
    wget \
    openssh-server

# Install GLOG (required by ceres).
RUN git clone --branch v0.6.0 https://github.com/google/glog.git --single-branch && \
    cd glog && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install && \
    cd ../.. && \
    rm -r glog
# Add glog path to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

# Install Ceres-solver (required by colmap).
RUN git clone --branch 2.1.0 https://ceres-solver.googlesource.com/ceres-solver.git --single-branch && \
    cd ceres-solver && \
    git checkout $(git describe --tags) && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j && \
    make install && \
    cd ../.. && \
    rm -r ceres-solver

# Install colmap.
RUN git clone --branch 3.7 https://github.com/colmap/colmap.git --single-branch && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install && \
    cd ../.. && \
    rm -r colmap

# Upgrade pip and install packages.
RUN pip install --upgrade pip setuptools pathtools promise && \
    pip install determined==0.23.4 && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Install tynyCUDNN.
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch

RUN pip install \
    nerfacc==0.3.5 \
    Pillow==9.5.0 \
    trimesh \
    natsort