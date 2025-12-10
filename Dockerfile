# Use the NVIDIA CUDA and cuDNN development image as the base
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Update package list and install Python, Git, and essential build tools
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common \
    python3 \
    python3-pip \
    python3-venv \
    git \
    build-essential \
    wget \
    curl \
    slurm-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link from python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create a virtual environment
ENV VENV_PATH="/opt/torch"

WORKDIR /workspace

# COPY . FastGraphCompute

RUN python -m venv $VENV_PATH && \
    . $VENV_PATH/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 && \
    pip install torch_geometric && \
    pip install git+https://github.com/jkiesele/FastGraphCompute && \
    pip install lightning wandb click seaborn atlasify scikit-learn && \
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html && \
    pip install scienceplots jsonargparse[signatures]>=4.27.7 rich && \
    # pip install -r FastGraphCompute/acorn.txt && \
    pip cache purge --no-input
 
ENV PATH="$VENV_PATH/bin:$PATH"



