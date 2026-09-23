# UAV Detection Dataset Preparation and Model Training

This repository documents the dataset preparation and training techniques I used while iteratively training UAV detection models.

Rather than training from a single fixed dataset, I combined multiple UAV datasets, trained successive models, inspected failure cases, and modified the training data based on what the models were getting wrong.

# Dataset List

Below is a non-comprehensive list of UAV datasets I used and explored in the process.

1. [Purdue Multi-Target UAV Detection and Tracking Dataset](https://engineering.purdue.edu/~bouman/UAV_Dataset/) — Air-to-air UAV video dataset with annotated target drones, including multi-UAV scenes and small aerial targets.

2. [ARD100 / YOLOMG](https://github.com/Irisky123/YOLOMG) — Air-to-air drone detection dataset used for UAV detection research, with associated training and dataset-processing code.

3. [Det-Fly](https://github.com/Jake-WU/Det-Fly) — Air-to-air UAV detection dataset containing thousands of images across sky, urban, field, and mountain backgrounds, with many small and distant targets.

4. [UAV Dataset on Zenodo](https://zenodo.org/records/7477569) — Annotated UAV image dataset captured under varied viewing angles and lighting conditions, with labels available in common detection formats.

5. [Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV) — UAV detection and tracking benchmark with RGB and thermal/infrared video, including small targets, dynamic backgrounds, and bounding-box annotations.

6. [Multi-View Drone Tracking Datasets](https://github.com/CenekAlbl/drone-tracking-datasets) — Multi-camera drone tracking datasets with 2D annotations and 3D trajectory ground truth, including fast motion, moving backgrounds, and multi-drone scenarios.

7. [TIB-Net Drone Dataset](https://github.com/kyn0v/TIB-Net) — Drone detection dataset released with TIB-Net, organized in VOC-style annotations and used for training/testing a lightweight drone detector focused on small aerial targets.

## Dataset Combination

I initially combined several UAV detection datasets into a single COCO-style dataset.

This required keeping image metadata, bounding boxes, category IDs, and annotation IDs consistent across sources so the resulting dataset could be used directly for RF-DETR training.

As the project progressed, I continued creating new versions of the combined dataset rather than treating the original dataset as fixed.

## Synthetic Copy-Paste Augmentation

One technique I introduced was offline copy-paste augmentation.

The augmentation script extracts annotated UAV crops from the existing COCO dataset and pastes them into other images to generate additional synthetic scenes. The generated images and annotations are written to separate folders so the original dataset is left unchanged. 

The script:
- builds a bank of UAV crops from annotated images
- selects UAV crops from other images
- rescales them relative to the UAV sizes already present in the target image
- finds valid placements that do not overlap existing bounding boxes
- pastes the UAV into the image
- automatically creates a new COCO bounding-box annotation

This was mainly intended to increase the number and variety of multi-UAV scenes instead of relying entirely on naturally occurring examples.

## Hard-Negative Mining

Another technique I used was hard-negative mining.

After training a model, I ran inference on additional imagery and reviewed high-confidence false detections. These false positives were collected and added as negative training examples for future models.

This made later datasets increasingly targeted toward actual model failure modes rather than simply increasing dataset size.

## Files

### `copy_paste_augment.py`

Generates synthetic multi-UAV COCO data using copy-paste augmentation.

The script extracts UAV crops from the original dataset, randomly rescales and places them into other images, avoids overlapping existing objects, and automatically generates new COCO annotations. :contentReference[oaicite:5]{index=5}

### `train.py`

Standard RF-DETR training script used to train detection models on the prepared datasets.
