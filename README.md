# YOLOv10 OpenVINO C++ Inference

Implementing YOLOv10 object detection using OpenVINO for efficient and accurate real-time inference in C++.

## Features
- [x] Support for `ONNX` and `OpenVINO IR` model formats
- [x] Support for `FP32` and `INT8` precisions

Tested on Ubuntu `18.04`, `20.04`, `22.04`.

## Dependencies
| Dependency | Version  |
| ---------- | -------- |
| OpenVINO   | >=2023.3 |
| OpenCV     | >=3.2.0  |
| C++        | >=14     |
| CMake      | >=3.10.2 |

## Model Conversion Resources
- [Docs by Ultralytics](https://docs.ultralytics.com/integrations/openvino/#usage-examples)
- [Supported models by OpenVINO](https://docs.openvino.ai/2023.3/openvino_docs_OV_UG_Integrate_OV_with_your_application.html)
- [YOLOv10 exporter](YOLOv10_exporter.ipynb)

## Installation Options

You have two options for setting up the environment: manually installing dependencies or using Docker.

<details>
  <summary>Option 1: Manual Installation</summary>

#### Install Dependencies
```bash
apt-get update
apt-get install -y \
    libtbb2 \
    cmake \
    wget \
    libopencv-dev \
    pkg-config \
    g++ \
    gcc \
    libc6-dev \
    make \
    build-essential \
    sudo \
    ocl-icd-libopencl1 \
    python3 \
    python3-venv \
    python3-pip \
    libpython3.8
```

#### Install OpenVINO
You can download OpenVINO from [here](https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux).
```bash
wget -O openvino.tgz https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64.tgz && \
sudo mkdir /opt/intel
sudo mv openvino.tgz /opt/intel/
cd /opt/intel
sudo tar -xvf openvino.tgz
sudo rm openvino.tgz
sudo mv l_openvino* openvino
```
</details>

<details>
  <summary>Option 2: Using Docker</summary>

#### Building the Docker Image
To build the Docker image yourself, use the following command:
```bash
docker build . -t yolov10
```

#### Pulling the Docker Image
Alternatively, you can pull the pre-built Docker image from Docker Hub (available for Ubuntu 18.04, 20.04, and 22.04):
```bash
docker pull rlggyp/yolov10:18.04
docker pull rlggyp/yolov10:20.04
docker pull rlggyp/yolov10:22.04
```

For detailed usage information, please visit the [Docker Hub repository page](https://hub.docker.com/repository/docker/rlggyp/yolov10/general).

#### Running a Container
To run a container from the image, use the following `docker run` command:

```bash
docker run -it --rm --mount type=bind,src=$(pwd),dst=/repo \
    --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -w /repo \
    rlggyp/yolov10:<tag>
```

</details>

## Build 
```bash
git clone https://github.com/rlggyp/YOLOv10-OpenVINO-CPP-Inference.git
cd YOLOv10-OpenVINO-CPP-Inference/yolo

mkdir build
cd build
cmake ..
make
```

## Usage
Yo can download the YOLOv10 model from here: [ONNX](https://github.com/rlggyp/YOLOv10-OpenVINO-CPP-Inference/raw/model/assets/yolov10n.onnx), 
[OpenVINO IR FP32](https://github.com/rlggyp/YOLOv10-OpenVINO-CPP-Inference/raw/model/assets/yolov10n_fp32_openvino.zip), 
[OpenVINO IR INT8](https://github.com/rlggyp/YOLOv10-OpenVINO-CPP-Inference/raw/model/assets/yolov10n_int8_openvino.zip)
```bash
# Run this command if you are using an ONNX model format
./detect <model_path.onnx> <image_path> 
# Or
# Run this command if you are using an OpenVINO IR model format
./detect <model_path.xml> <image_path> 
```
![result_bus](assets/result_bus.png)
![result_zidane](assets/result_zidane.png)

## Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.
