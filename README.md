# Multi-Camera Markerless Gait Analysis for Knee Osteoarthritis Research

This project is a research-oriented, end-to-end system for markerless, easily reproducible human gait analysis, developed in collaboration between the **University of Malaga** and **Costa del Sol Hospital**. It combines multi-camera computer vision, 2D human pose estimation, geometric 3D reconstruction, and biomechanical analysis to study gait patterns associated with knee osteoarthritis.

## Project Architecture

The system is divided into two complementary repositories:

- **markerless-gait-analysis-backend**: processes the multi-camera videos, detects 2D keypoints, combines the outputs of several pose-estimation models, reconstructs the human pose in 3D, and computes analysis measures.
- **markerless-gait-analysis-frontend**: captures and records the synchronized camera streams and sends the video chunks to the server.

Together, they provide the complete workflow for acquiring and analyzing gait data in a clinical or research setting. This repository documents the server component; both repositories are required for a complete deployment.

## Technical Overview

The server receives video chunks from several cameras and executes the following pipeline:

1. Extract frames from the incoming video chunks.
2. Detect 2D human-pose keypoints independently with several MMPose models.
3. Fuse the detector outputs using a confidence-weighted ensemble.
4. Estimate the spatial configuration of the cameras from multi-view keypoint correspondences.
5. Triangulate the 2D keypoints to obtain an initial 3D reconstruction.
6. Refine the camera parameters and 3D keypoints jointly through bundle adjustment.
7. Compute reprojection errors, joint angles, anthropometric measures, and other analysis outputs.

This organization separates perception, fusion, geometry, optimization, and analysis. It also makes it possible to inspect the contribution of each model and each reconstruction stage rather than relying only on a final aggregate result.

## Pose-Estimation Models

The system currently supports the following MMPose-based detectors:

- **ViTPose**: COCO, 17 keypoints.
- **HRNet**: WholeBody, 133 keypoints covering the body, feet, hands, and face.
- **CSPNeXt**: WholeBody, 133 keypoints covering the body, feet, hands, and face.
- **MSPN**: supported by the detector architecture and configuration included in the project.

Each detector is implemented independently, with its own model weights and keypoint definition. New MMPose detectors can be added by inheriting from `backend/processing/detectors/base.py`. External models can also be integrated by implementing or overriding the required initialization and chunk-processing methods.

Model checkpoints are distributed through a project release and must be placed at:

```text
Server/mmpose_models/checkpoints/
```

For a detailed description of the main backend classes and methods, see `docs/main_classes.md`.

## Confidence-Weighted Ensembling

The pose-processing coordinator abstracts the execution of multiple 2D pose detectors. It initializes the selected detectors when the first chunk arrives, distributes work across the available GPUs, and processes every chunk with all active models.

For each keypoint, the ensemble combines the detector predictions using two sources of information:

- the confidence reported by each detector;
- a detector-specific weight assigned to the keypoint.

The final coordinates and confidence values are computed as weighted combinations of the available predictions. Ensembling begins asynchronously once all cameras have completed processing the corresponding chunk, allowing the multi-camera workflow to continue while preserving synchronization between views.

## 3D Reconstruction Pipeline

The reconstruction module uses a geometry-based pipeline designed to make the estimation process explicit and evaluable.

### Camera geometry

The initial extrinsic parameters of the cameras are estimated from 2D keypoint correspondences across multiple views. This establishes the spatial relationship between the cameras and provides the geometric basis for reconstruction.

### Initial triangulation

Given the camera parameters and the fused 2D observations, the system reconstructs each anatomical keypoint in 3D using SVD-based triangulation.

### Bundle adjustment

The initial camera parameters and 3D points are jointly refined with nonlinear bundle adjustment. The optimization minimizes the global reprojection error, improving both the camera configuration and the spatial positions of the reconstructed keypoints.

### Anatomical and biomechanical analysis

The system applies anatomical scaling based on body measurements, including the nose-to-ankle distance, and provides analysis tools for:

- knee-flexion angles;
- body and anthropometric measurements;
- per-camera 2D keypoint observations;
- initial and optimized 3D reconstructions;
- reprojection errors for each camera and reconstruction method.

The 3D results are generated automatically after ensembling and stored in `data/processed/3D_keypoints/` using the `{frame_id}_{chunk_id}.npy` format.

## Repository Structure

```text
Server/
├── app.py
├── main.py
├── config/
│   ├── settings.py
│   ├── camera_intrinsics.py
│   └── __init__.py
├── backend/
│   ├── processing/
│   │   ├── ensemble/
│   │   │   └── ensemble_processor.py
│   │   ├── detectors/
│   │   │   ├── base.py
│   │   │   ├── vitpose.py
│   │   │   ├── mspn.py
│   │   │   ├── hrnet.py
│   │   │   └── csp.py
│   │   ├── reconstruction/
│   │   │   ├── camera.py
│   │   │   ├── calculate_extrinsics.py
│   │   │   ├── triangulation_svd.py
│   │   │   ├── bundle_adjustment.py
│   │   │   ├── perform_reconstruction.py
│   │   │   ├── reprojection.py
│   │   │   ├── analyze_3D_keypoint.py
│   │   │   └── complete_analysis.py
│   │   └── coordinator.py
│   └── tests/
├── mmpose_models/
│   ├── configs/
│   │   └── pose2d/
│   └── checkpoints/
├── data/
│   ├── unprocessed/
│   ├── processed/
│   └── logs/
├── docs/
│   └── main_classes.md
├── LICENSE.md
└── requirements.txt
```

The processing data is organized by patient, session, camera, detector, chunk, and frame. This preserves the provenance of the results and makes it possible to compare model predictions, ensemble outputs, and reconstruction stages.

## API

The server exposes the following endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/session/start` | Initialize a recording and processing session. |
| `POST` | `/api/chunks/receive` | Receive and process a video chunk from the client. |
| `POST` | `/api/session/end` | End the recording phase and determine the final chunk. |
| `POST` | `/api/session/cancel` | Cancel a session and remove its processed data. |
| `GET` | `/api/session/status` | Query the current session status. |
| `GET` | `/health` | Check server health. |
| `POST` | `/api/cameras/recalibrate` | Recalibrate the cameras' extrinsic parameters. |

Only one recording session can be active at a time, while processing can continue for multiple completed sessions.

## Running the Server

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Configure the environment in `config/settings.py`, including the server port, model paths, GPU allocation, and processing options. Then start the server:

```bash
python main.py
```

The default port is `5000`. When using the Client repository, its server address and port must match this configuration.

To generate visual demos of the per-camera 2D detections, set `save_annotated_videos = True` in `config/settings.py`. For this mode, the project recommends limiting `available_gpus` to a single GPU.

## Development and Testing

The `backend/tests/` directory contains focused prototypes and manual test scripts for detectors, video processing, 2D reconstruction, and 3D reconstruction. It is intended for developing and validating isolated components before integrating changes into the main pipeline rather than as a fully automated test suite.

## Research and Clinical Context

This system was developed from a real collaboration between a university and a hospital, where computer-vision methods had to be connected to practical clinical requirements. Its design reflects that setting: intermediate outputs are retained, reconstruction quality can be inspected through reprojection error, and the workflow supports communication between engineers, researchers, and medical professionals.

The system is intended for research and clinical evaluation support. Its outputs should be interpreted by qualified professionals and are not, by themselves, a medical diagnosis.

## License

This project is licensed under the Apache License 2.0. The included MMPose models and configurations are also distributed under Apache License 2.0. See `LICENSE.md` for the full terms and the recommended academic citation for MMPose.

Developed by the **University of Malaga** and **Costa del Sol Hospital**.
