FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update -y
RUN apt-get update -y
RUN apt install -y g++ make wget unzip vim htop git dstat
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt update
RUN apt-get install -y build-essential
RUN apt-get install -y libc-dev

# install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.5/cmake-3.31.5-linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh cmake-3.31.5-linux-x86_64.sh --skip-license --prefix=/opt/cmake/
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN apt install -y ninja-build

# Install nsight compute
RUN apt-get install -y nsight-compute-2024.1.1

RUN mkdir -p /working_dir
WORKDIR /working_dir
RUN git config --global --add safe.directory "*"