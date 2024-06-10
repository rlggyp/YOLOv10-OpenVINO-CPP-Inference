# YOLOv10 OpenVINO C++ Inference

Implementing YOLOv10 object detection using OpenVINO for efficient and accurate real-time inference in C++.

## Features
- [x] Support for `ONNX` and `OpenVINO IR` model formats
- [x] Support for `FP32`, `FP16` and `INT8` precisions

Tested on Ubuntu `18.04`, `20.04`, `22.04`.

## Dependencies
| Dependency | Version  |
| ---------- | -------- |
| OpenVINO   | >=2023.3 |
| OpenCV     | >=3.2.0  |
| C++        | >=14     |
| CMake      | >=3.10.2 |

## Installation Options

You have two options for setting up the environment: manually installing dependencies or using Docker.

<details>
  <summary><b>Manual Installation</b></summary>

#### Install Dependencies
```bash
apt-get update
apt-get install -y \
    libtbb2 \
    cmake \
    make \
    git \
    libyaml-cpp-dev \
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
  <summary><b>Using Docker</b></summary>

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
Grant the Docker container access to the X server by running the following command:
```bash
xhost +local:docker
````
To run a container from the image, use the following `docker run` command:

```bash
docker run -it --rm --mount type=bind,src=$(pwd),dst=/repo \
    --env DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev:/dev \
    -w /repo \
    rlggyp/yolov10:<tag>
```

</details>

## Build 
```bash
git clone https://github.com/rlggyp/YOLOv10-OpenVINO-CPP-Inference.git
cd YOLOv10-OpenVINO-CPP-Inference/src

mkdir build
cd build
cmake ..
make
```

## Usage
Yo can download the YOLOv10 model from here: [ONNX](https://github.com/rlggyp/YOLOv10-OpenVINO-CPP-Inference/raw/model/assets/yolov10n.onnx), 
[OpenVINO IR FP32](https://github.com/rlggyp/YOLOv10-OpenVINO-CPP-Inference/raw/model/assets/yolov10n_fp32_openvino.zip), 
[OpenVINO IR FP16](https://github.com/rlggyp/YOLOv10-OpenVINO-CPP-Inference/raw/model/assets/yolov10n_fp16_openvino.zip), 
[OpenVINO IR INT8](https://github.com/rlggyp/YOLOv10-OpenVINO-CPP-Inference/raw/model/assets/yolov10n_int8_openvino.zip)
### Using an ONNX Model Format
```bash
# For video input: 
./video <model_path.onnx> <video_path>
# For image input: 
./detect <model_path.onnx> <image_path>
# For real-time inference with a camera: 
./camera <model_path.onnx> <camera_index>
```

### Using an OpenVINO IR Model Format
```bash
# For video input: 
./video <model_path.xml> <video_path>
# For image input: 
./detect <model_path.xml> <image_path>
# For real-time inference with a camera: 
./camera <model_path.xml> <camera_index>
```
<p align="center"> 
  <img alt="traffic_gif" src="assets/traffic.gif", width="80%">
  <img alt="result_bus" src="assets/result_bus.png", width="80%">
  <img alt="result_zidane" src="assets/result_zidane.png", width="80%">
</p>

## References
- [How to export the YOLOv10 model](https://github.com/THU-MIG/yolov10?tab=readme-ov-file#export)
- [Convert and Optimize YOLOv10 with OpenVINO](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/yolov10-optimization/yolov10-optimization.ipynb)
- [Exporting the model into OpenVINO format](https://docs.ultralytics.com/integrations/openvino/#usage-examples)
- [Model Export with Ultralytics YOLO](https://docs.ultralytics.com/modes/export/)
- [Supported models by OpenVINO](https://docs.openvino.ai/2023.3/openvino_docs_OV_UG_Integrate_OV_with_your_application.html#step-2-compile-the-model)
- [YOLOv10 exporter notebooks](notebooks/YOLOv10_exporter.ipynb)

## Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.
