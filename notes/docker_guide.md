# Docker Setup for PyMesh Post-Processing

## Why Docker?

PyMesh cannot be installed via pip and requires building from source with specific C++/CMake dependencies. The `pymesh/pymesh` Docker image ships a pre-built PyMesh, avoiding the build process entirely.

The post-processing script (`mesh_postprocessing.py`) does not use PyTorch or CUDA — it only needs PyMesh, Open3D, PyVista, SciPy, and NumPy. This means the Docker container does not need GPU access.

## Recommended Workflow

```
Host (venv with torch+CUDA)          Docker (pymesh)
─────────────────────────────         ──────────────────────────
1. Run main.py                        3. Run mesh_postprocessing.py
   ↓ fits surfaces, generates            ↓ clips meshes, extracts
     unclipped meshes                      topology
   ↓                                     ↓
2. Save meshes + clusters to disk     4. Output: clipped meshes + topo JSON
```

The fitting stage (step 1-2) runs on the host where PyTorch has CUDA access. The post-processing stage (step 3-4) runs in the Docker container where PyMesh is available. Intermediate data passes through the shared mounted directory.

## Installation

### Install Docker

```bash
sudo apt update
sudo apt install docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

Log out and back in for the group change to take effect. Verify with:

```bash
docker run hello-world
```

### Build the Container

From the project root (`point2cad_repr/`):

```bash
docker build -t point2cad-postprocess .
```

This uses the `Dockerfile` in the project root. The `.dockerignore` file excludes `venv/` and `__pycache__/` from the build context, so they are not copied into the image.

## Running the Container

### Interactive Session (Recommended for Development)

```bash
docker run -it -v $(pwd):/work point2cad-postprocess bash
```

- `-it`: interactive terminal
- `-v $(pwd):/work`: bind-mounts the current directory to `/work` inside the container
- The mount is a direct reference to the host filesystem — no data is copied, reads and writes go straight to disk
- Your venv directory will be visible at `/work/venv` but the container uses its own Python installation, so it is harmless

Inside the container:

```bash
cd /work
python mesh_postprocessing.py
```

### One-Shot Execution

```bash
docker run --rm -v $(pwd):/work point2cad-postprocess python /work/mesh_postprocessing.py
```

`--rm` removes the container after it exits.

## Notes

### Open3D Visualization

`o3d.visualization.draw_geometries()` will not work inside the container — it requires a display server. To visualize results, either:

1. Save meshes to `.ply` files inside the container, then view them on the host.
2. Forward X11 by adding `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` to the `docker run` command (requires `xhost +local:docker` on the host).

### Python Version Compatibility

The `pymesh/pymesh` image ships an older Python (likely 3.6-3.8). The post-processing script (`mesh_postprocessing.py`) uses only basic language features (f-strings, type hints) and is compatible with Python 3.6+. If you encounter version issues with Open3D or other packages, pin to older compatible versions in the Dockerfile:

```dockerfile
RUN pip install --no-cache-dir \
    "numpy<2" \
    "scipy<1.12" \
    "open3d==0.17.0" \
    "pyvista<0.43"
```

### CUDA in Docker (Not Required for Post-Processing)

If you later want to run the full pipeline (including fitting) inside Docker, you would need:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host
2. Use a CUDA-enabled base image instead of `pymesh/pymesh`
3. Build PyMesh from source on top of that image
4. Run with `docker run --gpus all ...`

This is significantly more complex. Since `mesh_postprocessing.py` is the only script that needs PyMesh and it does not use CUDA, the split workflow above avoids this complexity entirely.

### Persisting Container State

If you install additional packages inside an interactive session and want to keep them:

```bash
# After exiting the container, find its ID
docker ps -a

# Commit it as a new image
docker commit <container_id> point2cad-postprocess:custom
```

Alternatively, add the packages to the `Dockerfile` and rebuild.

## Full Pipeline Container (Point2CAD Approach)

The original Point2CAD builds PyMesh from source on top of an NVIDIA CUDA image, giving both CUDA and PyMesh in one container. This avoids the split workflow above and lets you run the entire pipeline (fitting + post-processing) inside Docker.

Reference: [`point2cad/build/Dockerfile`](../point2cad/build/Dockerfile), [`point2cad/build/entrypoint.sh`](../point2cad/build/entrypoint.sh)

### Prerequisites

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so Docker can access the GPU:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify with:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

### What the Dockerfile Does

The `Dockerfile` in the project root follows Point2CAD's build strategy:

1. **Base image**: `nvidia/cuda:12.6.3-devel-ubuntu22.04` — provides CUDA 12.6 development tools, matching the host CUDA version.

2. **System dependencies**: gcc, g++, cmake, libgmp, libmpfr, libboost, and other libraries required to compile PyMesh. Also installs `gosu` for the entrypoint's UID/GID mapping.

3. **Python 3.10**: Installed via the deadsnakes PPA. Python 3.10 is used instead of the project's 3.12 because PyMesh (last maintained ~2020) is more likely to compile against 3.10. The project code uses no Python 3.11+ features and runs on 3.10 without changes.

4. **PyMesh from source**: Clones PyMesh at commit `384ba882` (the same pinned commit Point2CAD uses), builds a wheel via `setup.py bdist_wheel`, patches it, and installs it. The build artifacts are cleaned up to reduce image size.

5. **PyTorch with CUDA 12.6**: Installed from PyTorch's cu126 wheel index. PyTorch bundles its own CUDA runtime libraries via pip, so it does not depend on the system CUDA installation at runtime — but the system CUDA devel tools were needed for the PyMesh build.

6. **Other dependencies**: numpy, scipy, open3d, pyvista, matplotlib, tqdm.

### The Entrypoint

`entrypoint.sh` solves the bind-mount file permission problem. When you mount a host directory with `-v`, files created inside the container are owned by root on the host. The entrypoint reads the host UID/GID from the mounted `/work` directory, creates a matching user inside the container, and runs the command as that user via `gosu`. This way files written to the mount have correct ownership on the host.

Set `DEBUG=1` to print diagnostic information (CUDA availability, user mapping):

```bash
docker run --rm --gpus all -e DEBUG=1 -v $(pwd):/work point2cad bash -c "python -c 'import pymesh; import torch; print(torch.cuda.is_available())'"
```

### Build

```bash
docker build -t point2cad .
```

The PyMesh build step takes a long time (20-40 minutes). Docker caches each layer, so subsequent builds that only change later steps (e.g., pip packages) skip the PyMesh compilation.

### Run

Interactive session with GPU access:

```bash
docker run -it --gpus all -v $(pwd):/work point2cad_repr bash

docker run --rm -it --gpus all -e DISPLAY=${DISPLAY} -v $(pwd):/work point2cad_repr bash
```

One-shot execution:

```bash
docker run --rm --gpus all -v $(pwd):/work point2cad_repr python /work/main.py
```

### Python Version Compatibility

The container uses Python 3.10 while the host venv uses Python 3.12. The project code is compatible with both — it uses no 3.11+ features (`match/case`, `ExceptionGroup`, `except*`, `type` aliases, or lowercase generic annotations at runtime). The only difference is that package versions inside the container may differ from the host (pip resolves versions compatible with 3.10).

## Docker commands
```
docker ps - lists active containers

docker image ls - lists images present on the local system

docker build -t <tag> . - Builds an image from the Dockerfile in the current directory with the given tag

docker run --rm -it --gpus all -e DISPLAY=${DISPLAY} -v $(pwd):/work <image_name> <command>
docker run --rm -it --gpus all -v $(pwd):/work <image_name> <command>
docker run --rm -it --gpus all -v $(pwd):/work point2cad_repr bash
```

### Point2CAD repr shortcuts
```
# Run the algortihm for a sample point cloud

# 2 planes + INR point cloud
python main.py --input ./sample_clouds/abc_00470.xyzc
python main.py --visualize --visualize_id 00470

# Planes surrounded by cylinder point cloud
python main.py --input ./sample_clouds/abc_00949.xyzc
python main.py --visualize --visualize_id 00949

# Cannon-like point cloud

# ABC dataset inspection and generating point clouds from it
python abc_preprocess.py --abc_dir ../../abc_dataset --model_id 00000077 --output_dir ../sample_clouds --num_points 30000 --min_points_per_surface 1500 --visualize
```