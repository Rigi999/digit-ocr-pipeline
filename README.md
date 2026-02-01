# Digit OCR Pipeline

This project implements a complete and educational OCR pipeline for handwritten digit recognition using PyTorch.

The main goal of this project is **learning**, not just achieving high accuracy.

## Project Goals
- Understand how an OCR pipeline works end-to-end
- Learn how data flows from raw images to predictions
- Build a clean and modular PyTorch project structure
- Practice GPU-based training with CUDA
- Prepare a professional GitHub-ready project

## Project Structure
digit-ocr-pipeline/
├── data/          # Datasets (e.g. MNIST)
├── models/        # Saved trained models (.pth)
├── notebooks/     # Experiments and analysis
├── reports/       # Plots and evaluation results
├── src/           # Source code
│   ├── config/    # Configuration files
│   ├── data/      # Data loading & preprocessing
│   ├── models/    # Model architectures
│   ├── train/     # Training logic
│   ├── inference/ # Prediction pipeline
│   └── utils/     # Helper functions
├── requirements.txt
├── .gitignore
└── README.md
