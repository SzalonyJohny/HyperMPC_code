FROM nvidia/cuda:12.5.1-devel-ubuntu22.04 as base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-tk \
    libomp-dev \
    vim \
    wget \
    bash-completion \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-utils \
    xvfb \
    x11-utils \
    x11-xserver-utils \
    libosmesa6-dev \
    patchelf \
    ffmpeg


# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# install requirements
COPY requirements.txt .
RUN pip install -U pip  \
    && pip install -r requirements.txt


# # Install acados
RUN cd /opt/ && git clone https://github.com/acados/acados.git
COPY . /opt/acados
RUN cd /opt/acados && git submodule update --recursive --init
RUN mkdir -p build 
WORKDIR /opt/acados/build
RUN cmake -DACADOS_WITH_QPDUNES=ON -DACADOS_WITH_OSQP=ON -DACADOS_WITH_QPOASES=ON -DACADOS_NUM_THREADS=8 -DACADOS_WITH_OPENMP=ON .. && make install -j8
RUN pip3 install -e /opt/acados/interfaces/acados_template
RUN wget https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux && chmod +x t_renderer-v0.0.34-linux && mv t_renderer-v0.0.34-linux /opt/acados/bin/t_renderer


# # install learning for casadi
RUN cd /opt/ && git clone https://github.com/Tim-Salzmann/l4casadi.git
COPY . /opt/l4casadi
RUN pip3 install -r /opt/l4casadi/requirements_build.txt
RUN pip3 install /opt/l4casadi/ --no-build-isolation


# install casadi nightly for acados sens eval
RUN mkdir -p /opt/casadi
WORKDIR /opt/casadi
RUN wget https://github.com/casadi/casadi/releases/download/nightly-main/casadi-3.6.7.dev+main-cp310-none-manylinux2014_x86_64.whl
RUN pip install casadi-3.6.7.dev+main-cp310-none-manylinux2014_x86_64.whl


# Clean up the package cache to reduce the image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*


RUN echo "export ACADOS_SOURCE_DIR=\"/opt/acados\"" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:\"/opt/acados/lib\"" >> ~/.bashrc

ENV PYTHONPATH=/hyper_prediction_models
WORKDIR /hyper_prediction_models
