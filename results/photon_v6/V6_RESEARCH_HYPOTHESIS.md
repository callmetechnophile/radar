# View-of-Delft (VoD) Phase V6 Research Hypothesis

## Primary Research Question
> *"Can temporal physics-aware radar representations learned from Oxford RobotCar transfer to 3D object detection and tracking on View-of-Delft through a dataset-specific radar adapter?"*

## Experimental Comparison Framework (Phase V6.1+)
- **System A**: VoD-only Temporal Mamba (trained from scratch on VoD)
- **System B**: Oxford V5.4 Zero-Shot Initialization + VoD Linear 3D Head
- **System C**: Oxford V5.4 Physics-Aware Foundation + Fine-Tuning on VoD

## Target Metrics:
1. 3D Object Detection mAP (Car, Pedestrian, Cyclist)
2. 3D Bounding-Box Center Translation Error (m)
3. Velocity & Yaw Heading Estimation Error
4. Multi-Frame Track Persistence & ID Switching Frequency
