# MLOps PyTorch Pipeline

An end-to-end MLOps pipeline for training and serving a PyTorch image classification model. The project uses CIFAR-10, configurable training parameters, Docker containerization, automated tests, and a REST API for model inference.

## Project Overview

This project implements a reproducible machine learning workflow consisting of:

1. PyTorch-based image classification
2. Configurable model training
3. Early stopping and checkpointing
4. Structured JSON-line metric logging
5. Automated model tests
6. Dockerized training workload
7. Dockerized model-serving application
8. REST-based prediction and health-check endpoints

> **Note:** Kubernetes deployment manifests are included in the repository structure but were not part of the completed local validation for this submission.

## Architecture

```text
                    ┌──────────────────────┐
                    │   CIFAR-10 Dataset   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    dataset.py        │
                    │ Transforms + Loader  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      model.py        │
                    │    CNN / ResNet      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      train.py        │
                    │ Config + Training    │
                    │ Early Stopping       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Model Checkpoint     │
                    │ checkpoints/*.pt     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       serve.py       │
                    │   REST API :8080     │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
          GET /health                 POST /predict
```

## Project Structure

```text
mlops-pytorch-pipeline/
│
├── README.md
├── .gitignore
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── src/
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   ├── serve.py
│   └── __init__.py
│
├── configs/
│   └── training_config.yaml
│
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
│
├── requirements/
│   ├── train.txt
│   └── serve.txt
│
├── tests/
│   ├── test_model.py
│   └── test_model_smoke.py
│
├── checkpoints/
├── data/
│
└── k8s/
    ├── namespace.yaml
    ├── configmap.yaml
    ├── training-job.yaml
    ├── serving-deployment.yaml
    ├── serving-service.yaml
    └── hpa.yaml
```

## Environment Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install training dependencies:

```bash
pip install -r requirements/train.txt
```

Install serving dependencies:

```bash
pip install -r requirements/serve.txt
```

## Configuration

Training parameters are stored in:

```text
configs/training_config.yaml
```

The configuration controls the model architecture, number of classes, number of epochs, batch size, learning rate, early-stopping patience, dataset location, and checkpoint output location.

This avoids hard-coding training parameters in the Python implementation.

## Training Locally

Run:

```bash
python src/train.py
```

Training metrics are printed to stdout in structured JSON-line format.

Example:

```json
{"epoch": 1, "train_loss": 1.82, "train_accuracy": 0.31}
{"epoch": 2, "train_loss": 1.42, "train_accuracy": 0.47}
```

The trained model is saved as a configurable checkpoint under the checkpoint directory.

## Running Tests

Run:

```bash
pytest -q
```

The test suite verifies the model implementation and performs a lightweight model smoke test.

## Docker Training

Build the training image:

```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
```

Run training with mounted data and checkpoint directories.

Linux/macOS:

```bash
docker run --rm \
-v $(pwd)/data:/app/data \
-v $(pwd)/checkpoints:/app/checkpoints \
mlops-train:v1
```

Windows PowerShell:

```powershell
docker run --rm `
-v ${PWD}/data:/app/data `
-v ${PWD}/checkpoints:/app/checkpoints `
mlops-train:v1
```

The training image uses a multi-stage Docker build and installs the pinned training dependencies.

## Docker Model Serving

Build the serving image:

```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

Run the serving application:

```bash
docker run --rm -p 8080:8080 \
-v $(pwd)/checkpoints:/app/checkpoints \
mlops-serve:v1
```

The serving container:

* Uses a slim Python base image
* Installs only inference dependencies
* Runs the application on port `8080`
* Runs as a non-root user
* Includes a Docker health check

## API Endpoints

### Health Check

```http
GET /health
```

Returns HTTP 200 when the model is successfully loaded.

Example:

```bash
curl http://localhost:8080/health
```

### Prediction

```http
POST /predict
```

Accepts an image using multipart form data and returns class probabilities.

Example:

```bash
curl -X POST http://localhost:8080/predict \
-F "image=@test_image.png"
```

## Git Workflow

The project follows a feature-branch workflow.

```text
main
  │
  ▼
develop
  │
  ├── feature/pytorch-model
  │       └── Pull Request #1
  │
  ├── feature/docker-training
  │       └── Pull Request #2
  │
  └── feature/<future-feature>
          └── Pull Request
```

All feature work is developed on feature branches and merged through Pull Requests.

Commit messages follow the Conventional Commits format, for example:

```text
feat: add model training pipeline
test: add CIFAR-10 model validation
docs: add Docker image metadata
fix: resolve model loading issue
```

## Technologies

* Python
* PyTorch
* Torchvision
* Flask/FastAPI
* Docker
* Pytest
* Git/GitHub
* Kubernetes manifests

## Submission Scope

The primary implementation and local validation for this submission covers:

* Repository setup
* PyTorch model implementation
* Dataset and training pipeline
* Model serving API
* Docker training image
* Docker serving image
* Automated tests

Kubernetes manifests are included for the extended deployment workflow but were not locally validated as part of the time-constrained implementation.
