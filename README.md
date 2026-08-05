# Depth Estimation

## About The Component

Capsule is a deep learning structure for predicting the depth (distance) of objects from a single 2D image or real-time video stream. The capsule input is an Image (`inputImage`), and outputs are a colored Depth Map Image (`outputDepthImage`) and Raw Depth Data array (`outputDepthArray`). The capsule utilizes state-of-the-art DepthAnything V2 and V3 models to draw highly accurate 3D depth layers using the Inferno colormap.

## Built With

The containers and libraries that work together with the capsule are as follows:

- PyTorch
- DepthAnything (V2 & V3)
- Redis

## Configs

- **Model Version** 
  - DepthAnythingV2 (Small, Base, Large)
  - DepthAnythingV3 (Small, Base, Large)
- **Device** 
  - CPU
  - GPU 

The **Model Version** configuration offers a selection between DepthAnything V2 and V3 architectures. You can choose one of them based on your performance needs.
- Small models (24M params) offer the fastest inference times, perfectly suitable for real-time video streaming.
- Base (97M) and Large (335M) models offer significantly more detailed and sharp depth boundaries at the cost of processing speed.

The **Device** configuration allows you to select the processing unit.

> **Warning**  
> If you select GPU, ensure that CUDA and appropriate PyTorch libraries are available on your system, otherwise the system will automatically fall back to CPU.

## Resources
List of the resources.

- Depth Anything V2 (GitHub)
- Depth Anything V3 (GitHub)
- OpenCV Colormaps (Inferno)

> E
