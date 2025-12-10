# 📝 README — Microfossil Detection Project

## 🌱 What is the microfossil detection problem?

Microfossils — tiny remains of ancient organisms — are essential for:

- geological dating  
- paleoenvironment reconstruction  
- reservoir analysis  
- academic research  

Traditionally, microfossil identification is **manual, slow and expert-dependent**.  
This project automates the process using **object detection and classification models** trained on microscope images annotated with tools like Label Studio.

Main challenges:

- inconsistent class names  
- noisy or incomplete bounding boxes  
- class imbalance  
- multiple taxonomic categories  
- multi-stage classification needs  

---

## 📁 Directory Structure
```bash
project/
│
├── train.py                       # Training entry point
├── validate.py                         # Optional launcher
├── dataset.py                     # Dataset builder / cross-validation generator
│
├── configs/
│   ├── training/
│   │   ├── full_cross_validation
│   │   ├── retinanet.yaml
│   │   ├── frcnn.yaml
│   │   └── ssdlite.yaml
│   └── dataset.yaml               # Dataset / CV configuration
│
├── experiments/                   # Auto-generated outputs
│   └── <model_name>/run_YYYY_MM_DD_HH_MM/
│       ├── config.yaml
│       ├── models/
│       │   ├── best.pth
│       │   └── last.pth
│       └── metrics.json
│
├── src/
│   ├── core/
│   │   └── config.py
│   │
│   ├── cross_validation/
│   │   └── cross_validator.py
│   │
│   ├── pipeline/
│   │   ├── dataset_pipeline.py
│   │   ├── training_pipeline.py
│   │   └── validate_pipeline.py
│   │
│   ├── trainer/
│   │   ├── frcnn_trainer.py
│   │   ├── retina_trainer.py
│   │   └── ssd_trainer.py
│   │
│   ├── dataset/
│   │   ├── coco_dataset.py
│   │   └── ssd_dataset.py
│   │
│   └── ingestion/
│       └── zip_loader.py
│
├── preprocess/
│   ├── canonical/
│   ├── normalization/
│   ├── transformer/
│   └── validation/
│       ├── dataset_builder.py
│       ├── parser.py
│       └── harmonizer.py
│
├── requirements_base.txt
├── requirements_torch.txt
│
├── Dockerfile
└── docker-compose.yml
```

## Useful Commands
### Build Docker Image
```bash
docker compose build
```


### Generate Dataset / Cross-Validation Splits
```bash
docker compose run marina python dataset.py --config config/dataset.yaml
```

### Run Full Cross-Validation
```bash
docker compose run marina python train.py --config config/training/full_cross_validation
```


### Train Models
### Train RetinaNet
```bash
docker compose run marina python train.py --config config/training/retinanet.yaml
```


### Train Faster R-CNN
```bash
docker compose run marina python train.py --config config/training/frcnn.yaml
```


### Train SSD Lite
```bash
docker compose run marina python train.py --config config/training/ssdlite.yaml
```


### ✅ Run Validation

Generate validation metrics and curves using an existing experiment:

Local
```bash
python validate.py --experiment run_2025_12_08_15_57
```

With Docker
```bash
docker compose run marina python validate.py --experiment run_2025_12_08_15_57
```

### Output Example (auto-generated)
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
