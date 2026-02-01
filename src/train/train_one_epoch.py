import torch
from torch.nn import CrossEntropyLoss


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = CrossEntropyLoss()
    total_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
