## 🌱 What is the microfossil detection problem?

Microfossils—tiny remains of ancient organisms—are essential for:

- geological dating

- paleoenvironment reconstruction

- reservoir analysis

- academic research

Traditionally, microfossil identification is manual, expert-dependent, and time-consuming.
This project aims to automate this process using object detection and classification models, trained over microscopy images annotated using tools like Label Studio.

Challenges include:

- inconsistent class names across experiments

- noisy or incomplete bounding boxes

- class imbalance

- multiple taxonomic categories

- multi-stage classification needs

## Directory Structure
```bash
project/
│
├── train.py # Entry point for training pipeline
├── run.py # Optional launcher for dataset or train routines
├── dataset.py # Dataset preparation (if needed)
│
├── configs/
│ ├── training/
|   ├── full_cross_validation
│   ├── retinanet.yaml
│   ├── frcnn.yaml
│   ├── ssdlite.yaml
│   └── dataset.yaml
│
├── experiments/ # Auto-generated logs / metrics / checkpoints
│ └── <model_name>/run_YYYY_MM_DD_HH_MM/
│ ├── config.yaml
│ ├── models/
│ │ ├── last.pth
│ │ └── best.pth
│ └── metrics.json
│
├── src/
│ ├── core/
│ │ └── config.py
│
│ ├── cross_validation/
│ │ └── cross_validator.py
│
│ ├── pipeline/
│ │ └── training_pipeline.py
│ │ └── training_pipeline.py
│ │
│ ├── trainer/
│ │ ├── frcnn_trainer.py
│ │ ├── retina_trainer.py
│ │ └── ssd_trainer.py
│ │
│ ├── dataset/
│ │ ├── coco_dataset.py
│ │ └── ssd_dataset.py
│ │
│ └── ingestion/ 
│   ├── zip_loader.py

│ ├── preprocess/
│ │ ├── canonical/
│ │ └── normalization/
│ │ ├── transformer/
│ │ └── validation/
│ │ ├── dataset_builder.py
│ │ ├── parser.py
│ │ └── harmonizer.py
│   
├── requirements_base.txt
├── requirements_torch.txt
│
├── Dockerfile
└── docker-compose.yml
```

## Usefull commands

## 

## Generate dataset
```bash
docker compose run marina python dataset.py --config configs/full_cross_validation
```
## Run Cross Validation
```bash
docker compose run marina python dataset.py --config configs/dataset.yaml
```

## Train RetinaNet
```bash
docker compose run marina python train.py --config configs/training/retinanet.yaml
```

## Train Faster R-CNN
```bash
docker compose run marina python train.py --config configs/training/frcnn.yaml
```


## Train SSD Lite
```bash
docker compose run marina python train.py --config configs/training/ssdlite.yaml
```
docker compose run marina python train.py --config configs/training/retinanet.yaml

## The result will be save at:

```bash
experiments/
    <model>/
        run_2025_01_01_12_30/
            config.yaml
            models/
                best.pth
                last.pth
            metrics.json
```