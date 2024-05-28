# YOLOv10 OpenVINO Inference C++
Implementing YOLOv8 object detection using OpenVINO for efficient and accurate real-time inference.

## Dependencies
| Dependency | Version  |
| ---------- | -------- |
| OpenVINO   | 2023.3   |
| OpenCV     | >=4.5.0  |
| C++        | >=14     |
| CMake      | >=3.12.0 |

## Model Conversion Resources
- [Docs by Ultralytics](https://docs.ultralytics.com/integrations/openvino/#usage-examples)
- [Supported models by OpenVINO](https://docs.openvino.ai/2023.3/openvino_docs_OV_UG_Integrate_OV_with_your_application.html)

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
```bash
./detect <model_path.onnx> <image_path.jpg>
```
## Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.
