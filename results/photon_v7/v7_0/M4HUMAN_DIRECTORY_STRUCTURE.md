# M4Human Dataset Directory Structure & Modality Inventory

## Dataset Root
C:\Users\worka\research\photonpinn\radar\datasets\m4human\

## Repository Architecture & Modality Layout

`
m4human/
├── config.yaml                          # Central dataset, model, and benchmark configuration
├── main1_multigpu_clean.py              # DDP distributed training & evaluation runner
├── data11_log_cali.txt                  # Extrinsic calibration logs (radar <-> camera <-> MoCap)
├── environment.yml                      # Conda environment definition (Python 3.9, PyTorch 2.2)
├── dataset/                             # Data loading and preprocessing pipelines
│   ├── m4human_dataset.py               # RF3DPoseDataset PyTorch Dataset implementation
│   ├── m4human_utils.py                 # File pair indexing, LMDB caching, and point extraction
│   ├── dataset_config_clean.py          # Split definitions (s1=random, s2=cross-subject, s3=cross-action)
│   ├── data_loader_camera_calibration.py# Extrinsic matrix transforms (Vicon -> RGB -> Radar)
│   ├── data_loader_Load_data.py         # JSON parameter & .MAT file loaders
│   ├── data_loader_Plotting_projection.py# 3D-to-2D projection & GIF visualization
│   └── lmdb_utils.py                    # MessagePack serialization & LMDB packing/unpacking
├── mmwave_models/                       # Neural model architectures
│   ├── Point_models/                    # P4Transformer point-cloud models
│   └── Tensor_models/                   # Radar Tensor (RT) architectures (RT-Mesh, RETR)
│       ├── RTmesh/                      # 2D BEV Transformer + 3D RoI Transformer
│       └── retr_models/                 # RETR radar encoder
├── sources/                             # Loss functions & metrics
│   ├── Train_and_model_loss.py          # SMPL-X loss definitions (betas, pose, trans, vertices)
│   ├── evaluation_module_pc_multigpu.py # MPJPE, PA-MPJPE, and 3D mesh evaluation metrics
│   └── Train_and_model_plotting_3D_mesh.py # 3D mesh plotting utilities
├── vis_depth/                           # Visual demonstration sequences
└── models/                              # Pretrained weights & SMPL-X body models
    └── smplx/                           # SMPLX_MALE, SMPLX_FEMALE, SMPLX_NEUTRAL body models
`

## Supported Modalities
1. Radar Point Cloud (RPC): Native single-frame point clouds (x, y, z, power, Doppler).
2. Radar Tensor (RT): 3D spatial range-azimuth-elevation power volume (121 x 111 x 31).
3. RGB Camera: Calibrated camera video streams (1920 x 1080).
4. Depth Camera: Calibrated depth maps for multimodal validation.
5. Vicon Motion Capture Ground Truth: 3D SMPL-X body parameters (22 joints, 3D translation, root orientation, shape betas, gender).
