print(">>> predict_digit.py STARTED <<<")

import torch
from torchvision import transforms
from PIL import Image

from src.models.cnn import SimpleCNN

def load_model(model_path, device):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    x = transform(img)
    x = x.unsqueeze(0)
    return x

def predict_digit(model, x, device):
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()

    return pred

def main():
    
    print(">>> main() ENTERED <<<")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("models/best_model.pth", device)


    image_path = "data/sample.png"
    x = preprocess_image(image_path)

    pred = predict_digit(model, x, device)
    print("Predicted digit:", pred)


if __name__ == "__main__":
    main()
