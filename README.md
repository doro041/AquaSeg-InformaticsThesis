
# User Manual

## Introduction

This user manual provides guidance for installing, configuring, and operating the AquaSeg mobile application for real-time detection and segmentation of egg-bearing lobsters. AquaSeg is designed for use by fisheries inspectors, marine biologists, conservationists, and fishers, enabling rapid, accurate identification of lobsters in field conditions using a standard Android mobile device.

## System Requirements

- Android device running Android 9.0 (API level 28) or higher
- Minimum 3GB RAM recommended
- At least 100MB free storage space
- Camera with autofocus capability

## Installation

1. **Download the source code** from the official GitHub repository and go to the MobileApp directory: [GitHub Repository](https://github.com/doro041/Pipeline/tree/main/MobileApp)
2. **Open the project** in Android Studio on your computer.
3. **Connect your Android device** via USB and enable developer mode and USB debugging.
4. **Build and run the project**:
   - Click "Run" in Android Studio to build the app and install it on your connected device.
   - Grant permission to access your camera.

## Application Overview

Upon launching AquaSeg, three main screens are accessible:

- **Introduction**: Overview of app features and intended use.
- **Camera**: Main screen for real-time detection and segmentation.
- **About**: Version, credits, and support information.

## Using the Camera Feature

![Camera screen showing real-time lobster detection and segmentation using YOLO11.](images/mobiledevelopment/yolo11.jpg)

1. Go to the Camera screen.
2. Point the camera at the lobster, ensuring the underside is visible.
3. The app will highlight detected lobsters with colored masks (blue for egg-bearing, red for general).
4. Use the pipeline toggle bar to switch between YOLO-only (faster) and YOLO+FastSAM (more detailed) segmentation.
5. Tap the capture button to save a snapshot of the current view and segmentation result.
6. Access saved images via your gallery.

## Best Practices

- Ensure good lighting and a clear view of the lobster’s underside.
- Hold the device steady for best results.
- Clean the camera lens regularly.
- Use the zoom feature for small or distant lobsters.

## Troubleshooting

- **Slow performance**: Close background apps or switch to YOLO-only mode.
- **Poor segmentation**: Improve lighting or reposition for better visibility.
- **App crashes**: Ensure your device meets requirements and try reinstalling or rebuilding the app.

## Support and Feedback

For support, bug reports, or feature requests, please open an issue on the [GitHub repository](https://github.com/doro041/Pipeline) or contact: `u17ds20@abdn.ac.uk`.

---

# Maintenance Manual

## Overview

This maintenance manual is intended for developers and researchers who wish to maintain, update, or extend the AquaSeg mobile application and its underlying machine learning pipelines. It covers both the software architecture and the machine learning workflows, including guidelines for integrating alternative models beyond YOLO.

## Disclaimer

Not all files and datasets referenced in this manual or described in the codebase structure are publicly available in the GitHub repository.

- **Datasets**: Original datasets, including annotated images and ground truth masks, are not included due to their large size.
- **Pretrained Model Weights**: Some trained model files are too large for GitHub or are subject to sharing restrictions.
- **Evaluation Results**: Evaluation outputs and result files are also too large to be uploaded to GitHub.

If you require access to files that are not present in the repository:

- Check the `README` for direct download links or additional instructions.
- Contact at `u17ds20@abdn.ac.uk`, or open a GitHub issue to request specific files.
- For public datasets, refer to the links provided in the thesis, `README`, or this manual.

## Codebase Structure

## Repository Structure Overview

The project directory contains both machine learning pipelines and the mobile application. Top-level folders:

- `datasets`: (Not included) Dataset of annotated images and masks.
- `evaluation_results`: Model evaluation outputs (qualitative and quantitative).
- `evaluation_seg`: Segmentation evaluation outputs and YOLO11-seg runs.
- `inference`: Trained models, inference scripts, and output runs.
- `preprocessing_data`: Data preparation and augmentation scripts.
- `runs`: YOLO object detection training runs and models.
- `MobileApp`: Android application source code and resources.

### Mobile Application Structure (`MobileApp/`)

- `.idea`, `gradle/`: Android Studio project configs.
- `app/src/main/assets/`: Model files (`.tflite`) and screen assets.
- `app/src/main/java/com/example/aquasegv2/ui/`: UI modules (about, camera, home).
- `app/src/main/res/`: Resources (layouts, icons, navigation).
- `app/src/androidTest/`, `test/`: Instrumented and unit tests.

---

## Machine Learning Pipeline Maintenance

### Training New Models

1. Prepare your dataset in `datasets/` following YOLO or COCO format.
2. Use `preprocessing_data/` for augmentation/preprocessing.
3. Train models in `inference/` or `runs/` with frameworks like Ultralytics.
4. Save weights and export them to ONNX or TensorFlow Lite (`.tflite`).
5. Store models in `inference/models/` and `MobileApp/app/src/main/assets/`.

### Evaluating Models

1. Use scripts in `evaluation_seg/` and `evaluation_results/` for evaluation (IoU, Dice, mAP).
2. Save figures, tables, and logs properly.
3. For visual comparison, add outputs to `comparison_direct/`, sorted by class and difficulty.

| Quality Level | Description | Visual Characteristics (Compared to Ground Truth) |
|:--------------|:------------|:--------------------------------------------------|
| High Quality | Prediction overlaps ground truth almost perfectly. | Precise boundaries, few false positives/negatives, tightly fit masks. |
| Medium Quality | Prediction mostly correct with minor errors. | Minor boundary errors, slight over/under-segmentation. |
| Low Quality | Prediction poorly matches ground truth. | Large areas missed, wrong masks, lots of noise/gaps. |

---

## Android Application Maintenance

### Updating or Replacing Models

1. Add new `.tflite` models to `MobileApp/app/src/main/assets/`.
2. Update Java/Kotlin files for label parsing, output shape, etc., if needed.
3. Rebuild and test the app on devices.

### Development and Testing

- Open `MobileApp/` in Android Studio.
- UI logic is modularized (`ui/about`, `ui/camera`, `ui/home`).
- Layouts and navigation are under `res/`.
- Build and run the app on device or emulator.
- Update/add tests in `androidTest/` and `test/`.

---

## Extending and Experimenting with ML Models

### Guidelines for Alternative Models

- Ensure the model can export to `.tflite`.
- Adapt input/output mapping to fit the app’s pipeline.
- Document any changes for reproducibility.
- Profile model speed and memory usage before full deployment.

### Dataset Expansion

- Add new annotated images to `datasets/`.
- Use `preprocessing_data/` for augmentation.
- Update data splits accordingly.

### Ground Truth Fixing

- Improve masks using Roboflow for fine-grained annotations.
- Apply augmentations using `preprocessing_data/`.
- Ensure consistent naming with model outputs.

---

## Performance and Resource Optimization

- Quantize models (e.g., Float16) to improve mobile performance.
