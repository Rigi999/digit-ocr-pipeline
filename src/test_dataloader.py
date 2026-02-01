from src.data.dataloader import create_dataloaders


train_loader, test_loader = create_dataloaders(batch_size=64)

images, labels = next(iter(train_loader))

print("images shape:", images.shape)
print("labels shape:", labels.shape)
print("first 10 labels:", labels[:10].tolist())
