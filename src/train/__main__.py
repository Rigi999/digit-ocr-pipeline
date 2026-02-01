import torch

from src.data.dataloader import create_dataloaders
from src.models.cnn import SimpleCNN
from src.train.train_one_epoch import train_one_epoch
from src.train.evaluate import evaluate_accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, test_loader = create_dataloaders(batch_size=64)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 3
    best_acc = 0.0
    best_path = "models/best_model.pth"

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device
        )

        test_acc = evaluate_accuracy(model, test_loader, device)

        print(f"Epoch {epoch}/{num_epochs} - train_loss: {train_loss:.4f}")
        print(f"Test Accuracy: {test_acc * 100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model (test_acc={best_acc * 100:.2f}%)")


if __name__ == "__main__":
    main()

