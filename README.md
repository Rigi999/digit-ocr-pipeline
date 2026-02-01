\# Digit OCR Pipeline (MNIST) — PyTorch



An end-to-end handwritten digit recognition pipeline built with \*\*PyTorch\*\*.

This project covers data loading, CNN training, evaluation, saving the best model checkpoint, and inference on an input image.



\## Highlights

\- Clean and modular project structure

\- GPU (CUDA) support if available

\- Automatic saving of the best-performing model

\- Simple inference pipeline for real images



\## Results

\- \*\*Test Accuracy:\*\* ~\*\*97.7%\*\* (after 3 epochs)

\- \*\*Example inference:\*\* `Predicted digit: 7`



\## Project Structure

```

digit-ocr-pipeline/

├── data/

│   └── sample.png

├── models/

│   └── best\_model.pth

├── notebooks/

├── reports/

├── src/

│   ├── data/

│   │   ├── mnist\_dataset.py

│   │   └── dataloader.py

│   ├── models/

│   │   └── cnn.py

│   ├── train/

│   │   ├── \_\_main\_\_.py

│   │   ├── train\_one\_epoch.py

│   │   └── evaluate.py

│   └── inference/

│       ├── create\_sample.py

│       └── predict\_digit.py

├── requirements.txt

└── README.md

```



\## Setup



\### Conda environment (recommended)

```bash

conda create -n ai python=3.11 -y

conda activate ai

pip install -r requirements.txt

```



\### GPU check

```bash

python -c "import torch; print(torch.cuda.is\_available())"

```



\## How to Run



\### Train and evaluate (saves best model automatically)

```bash

python -m src.train

```



The best checkpoint is saved to:

```

models/best\_model.pth

```



\### Create a sample input image (from MNIST)

```bash

python -m src.inference.create\_sample

```



This creates:

```

data/sample.png

```



\### Inference

```bash

python -m src.inference.predict\_digit

```



\## Model Architecture (SimpleCNN)

\- Input: `1 × 28 × 28` (grayscale)

\- Conv2D: `1 → 16`, kernel `3×3`, padding `1`

\- ReLU

\- MaxPool `2×2`

\- Flatten: `16 × 14 × 14 = 3136`

\- Linear: `3136 → 10` (logits)



\## Future Improvements

\- Add deeper CNN blocks

\- Add confusion matrix and evaluation reports

\- Add CLI arguments for training (epochs, lr, batch size)

\- Improve generalization with data augmentation



