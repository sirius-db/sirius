FROM nvidia/cuda:12.6.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root

# Common dependencies
RUN apt-get update && apt-get install -y git g++ cmake ninja-build libssl-dev wget vim unzip

# CMake 3.30.4
RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.4/cmake-3.30.4-linux-x86_64.sh && \
    chmod +x cmake-3.30.4-linux-x86_64.sh && \
    ./cmake-3.30.4-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.30.4-linux-x86_64.sh

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create conda env with libcudf
RUN conda update -n base -c defaults conda && \
    conda create -n libcudf-env -y && \
    conda run -n libcudf-env conda install -y -c rapidsai -c conda-forge -c nvidia rapidsai::libcudf
ENV USE_CUDF=1
ENV LIBCUDF_ENV_PREFIX=$CONDA_DIR/envs/libcudf-env

# Activates libcudf env at runtime beginning
RUN echo 'echo "[INFO] Initializing conda environment... please wait (conda activation can take a few seconds)"' >> ~/.bashrc && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate libcudf-env" >> ~/.bashrc && \
    echo 'echo "[INFO] Finished initializing conda environment"' >> ~/.bashrc