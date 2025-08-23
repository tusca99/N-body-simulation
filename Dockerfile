# Usa una base CUDA ufficiale con toolchain recente e compatibile
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Installa CMake >=3.25.2
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.sh && \
    sh cmake-3.28.3-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.28.3-linux-x86_64.sh

# Installa tool di build e librerie necessarie (OpenGL, GLFW, GLEW, ecc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libglew-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Imposta la directory di lavoro
WORKDIR /workspace

# Copia tutto il progetto nella directory di lavoro
COPY . .

# Build automatica all'avvio del container
ENTRYPOINT ["/bin/bash", "-c", "\
  rm -rf build && \
  chmod 777 build || true && \
  mkdir -p build && cd build && \
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS='-O3 --expt-relaxed-constexpr -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -cudart static' \
    -DCMAKE_CUDA_RUNTIME_LIBRARY=STATIC \
  && make -j$(nproc) n-body-simulation \
  && cp n-body-simulation .. \
  && echo 'Build completata. Eseguibile in /workspace/n-body-simulation' \
"]