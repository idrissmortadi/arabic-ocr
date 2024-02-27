import torch
import os
from datetime import datetime


def save_model_with_timestamp(model, folder_path="saved_models", suffix_format="%Y-%m-%d_%H-%M-%S"):
    """
    Saves the PyTorch model with a timestamp suffix to the specified folder.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        folder_path (str): The folder path where the model will be saved.
        suffix_format (str): The format of the timestamp suffix.
    """
    # Get the current date and time
    current_time = datetime.now().strftime(suffix_format)

    # Generate the file path with the timestamp suffix
    file_path = f"{folder_path}/{model.name}_{current_time}.pth"

    # Save the model
    torch.save(model, file_path)
    print(f"Model saved to {file_path}")


def find_size(numbers):
    first_non_zero_index = None
    last_non_zero_index = None

    for i, num in enumerate(numbers):
        if num != 0:
            if first_non_zero_index is None:
                first_non_zero_index = i
            last_non_zero_index = i

    # Check if any non-zero element found
    if first_non_zero_index is not None:
        size = last_non_zero_index - first_non_zero_index
    else:
        # If no non-zero elements found, set size to 0
        size = 0

    return {"start": first_non_zero_index,
            "end": last_non_zero_index,
            "size": size}


def square_crop(image, resize):
    # Find largest dimension vertical and horizontal
    v_dim = find_size(image.sum(axis=1))
    h_dim = find_size(image.sum(axis=0))

    # Create new image with largest dimension
    if h_dim["size"] > v_dim["size"]:
        cropped_image = torch.zeros(max(h_dim["size"]+1, 28),
                                    max(h_dim["size"]+1, 28))
    else:
        cropped_image = torch.zeros(max(v_dim["size"]+1, 28),
                                    max(v_dim["size"]+1, 28))

    # Cut the image and put it in the new cropped image then resize
    height, width = cropped_image.shape
    position_h = (height - v_dim["size"])//2, (height + v_dim["size"])//2
    position_w = (width - h_dim["size"])//2, (width + h_dim["size"])//2

    cropped_image[position_h[0]:position_h[1], position_w[0]:position_w[1]] =\
        torch.Tensor(
        image[v_dim["start"]:v_dim["end"], h_dim["start"]:h_dim["end"]])

    cropped_image = resize(cropped_image[None, ...])

    return cropped_image[0]


def get_unique_targets(labels_folder):
    unique_targets = set()

    for filename in os.listdir(labels_folder):
        label_path = os.path.join(labels_folder, filename)
        with open(label_path, 'r') as f:
            target = f.read()
            unique_targets.add(target.strip())

    return list(unique_targets)


def decode_target(id, unique_targets):
    return unique_targets[id]
