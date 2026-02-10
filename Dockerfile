FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt -y update --fix-missing && \
    apt install -y \
        gosu \
        gcc \
        g++ \
        gcc-9 \
        g++-9 \
        git \
        cmake \
        libgmp-dev \
        libmpfr-dev \
        libgmpxx4ldbl \
        libboost-dev \
        libboost-thread-dev \
        libgl1-mesa-glx \
        libxrender1 \
        libspatialindex-dev \
        software-properties-common \
        curl \
        zip \
        unzip \
        patchelf && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3.10-distutils && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    pip install --upgrade setuptools wheel

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-9

ENV ROOT_PYMESH=/opt/pymesh
RUN mkdir -p ${ROOT_PYMESH} && \
    cd ${ROOT_PYMESH} && \
    git init && \
    git pull https://github.com/PyMesh/PyMesh.git && \
    git checkout 384ba882 && \
    git submodule update --init && \
    sed -i '43s|cwd="/root/PyMesh/docker/patches"|cwd="'${ROOT_PYMESH}'/docker/patches"|' \
        ${ROOT_PYMESH}/docker/patches/patch_wheel.py && \
    pip install -r ${ROOT_PYMESH}/python/requirements.txt && \
    python setup.py bdist_wheel && \
    rm -rf build_3.10 third_party/build && \
    python ${ROOT_PYMESH}/docker/patches/patch_wheel.py dist/pymesh2*.whl && \
    pip install dist/pymesh2*.whl && \
    python -c "import pymesh; pymesh.test()"

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    open3d \
    pyvista \
    matplotlib \
    tqdm \
    trimesh

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

WORKDIR /work
