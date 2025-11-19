FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root

# Common dependencies
RUN apt-get update && apt-get install -y git g++ cmake ninja-build libssl-dev wget vim unzip

# CMake 4.1.0
RUN wget https://github.com/Kitware/CMake/releases/download/v4.1.0/cmake-4.1.0-linux-x86_64.sh && \
    chmod +x cmake-4.1.0-linux-x86_64.sh && \
    ./cmake-4.1.0-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-4.1.0-linux-x86_64.sh

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Accept TOS for anaconda channels
# RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
#    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
ENV ANACONDA_ACCEPT=1

# Create conda env with libcudf
RUN conda update -n base -c defaults conda && \
    conda create -n libcudf-env -y && \
    conda run -n libcudf-env conda install -y -c rapidsai -c conda-forge -c nvidia rapidsai::libcudf=25.10
ENV USE_CUDF=1
ENV LIBCUDF_ENV_PREFIX=$CONDA_DIR/envs/libcudf-env

# Activates libcudf env at runtime beginning
RUN echo 'echo "[INFO] Initializing conda environment... please wait (conda activation can take a few seconds)"' >> ~/.bashrc && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate libcudf-env" >> ~/.bashrc && \
    echo 'echo "[INFO] Finished initializing conda environment"' >> ~/.bashrc
