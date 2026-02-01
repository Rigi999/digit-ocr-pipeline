from torchvision import datasets
from PIL import Image
import os

# اطمینان از وجود پوشه data
os.makedirs("data", exist_ok=True)

# لود MNIST (test)
dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True
)

# گرفتن یک نمونه (مثلاً index=0)
img, label = dataset[0]

# ذخیره تصویر
img.save("data/sample.png")

print("Saved data/sample.png with label:", label)
