from torch.utils.data import random_split
from torch.utils.data import DataLoader


def create_data_loader(custom_dataset):
    # Define the sizes for train, validation, and test sets
    train_size = int(0.7 * len(custom_dataset))
    val_size = int(0.15 * len(custom_dataset))
    test_size = len(custom_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        custom_dataset,
        [train_size, val_size, test_size])

    # Example usage:
    print("Train set size:", len(train_dataset))
    print("Validation set size:", len(val_dataset))
    print("Test set size:", len(test_dataset))

    # Define batch size for train, validation, and test DataLoader instances
    batch_size_train = 32
    batch_size_val = 32
    batch_size_test = 32

    # Create DataLoader instances for train, validation, and test datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size_val,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size_test,
                             shuffle=False)

    # Example usage:
    print("Number of batches in train loader:", len(train_loader))
    print("Number of batches in validation loader:", len(val_loader))
    print("Number of batches in test loader:", len(test_loader))

    return train_loader, val_loader, test_loader
