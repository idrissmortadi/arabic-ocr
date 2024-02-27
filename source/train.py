from model_scripts import BenchmarkCnn2, HandwritingRecognizer1
import torch
import torch.nn as nn
from utils import save_model_with_timestamp
from custom_dataset import CustomDataset
from create_data_loader import create_data_loader
import torchvision

DATASET_IMAGES_PATH = "Dataset/Total Images/"
DATASET_LETTERS_PATH = "Dataset/Total GT/"

# Get train loaders
resize = torchvision.transforms.Resize((64, 64))
custom_dataset = CustomDataset(images_folder=DATASET_IMAGES_PATH,
                               labels_folder=DATASET_LETTERS_PATH,
                               transform=resize)
train_loader, val_loader, test_loader = create_data_loader(custom_dataset)

# Step 1: Define model

# Step 2: Set up your training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = len(custom_dataset.unique_targets)
model = HandwritingRecognizer1(64, 64, num_classes).to(device)
optimizer, lr_schedule = model.configure_optimizers()

criterion = nn.CrossEntropyLoss()

# Step 3: Train your model
num_epochs = 32

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        # print(model.features(data).shape)

        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'Step [{batch_idx + 1}/{len(train_loader)}]')
            print(f'Loss: {loss.item():.4f}')


# Save the trained model
save_model_with_timestamp(model)
