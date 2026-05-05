#!/usr/bin/env python3
"""
predict_my_digits.py

Predict digit images using a CNN trained on MNIST (improved_digit_cnn.pth).
Automatically inverts and normalizes images to match MNIST style.
"""

import argparse
import os
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from improved_digit_cnn import CNN


def image_to_mnist_tensor(path: str, device: torch.device, show=False):
    """
    Convert an input digit image into an MNIST-style tensor of shape (1, 1, 28, 28).
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    # Load image as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    # If background is light and digit is dark, invert it so digit becomes white on black
    if np.mean(img) > 127:
        img = 255 - img

    # Slight blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Threshold to make image binary
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find nonzero pixels to locate the digit
    coords = cv2.findNonZero(thresh)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = thresh[y:y+h, x:x+w]
    else:
        # Fallback: use the whole thresholded image
        digit = thresh

    # Pad to square
    h, w = digit.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = digit

    # Resize to 28x28 like MNIST
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

    # Convert to float in [0, 1]
    arr = resized.astype(np.float32) / 255.0

    # Normalize using same values as MNIST training
    arr = (arr - 0.1307) / 0.3081

    if show:
        cv2.imshow("Preprocessed", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Convert to tensor shape (1, 1, 28, 28)
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return tensor


def load_trained_model(model_path: str, device: torch.device):
    """
    Load the trained CNN model from a .pth file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = CNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_images(model_path: str, image_paths: List[str], device_str: str = "cpu", show=False):
    """
    Predict the digit in each image and return formatted output lines.
    """
    device = torch.device(device_str)
    model = load_trained_model(model_path, device)

    results = []

    for image_path in image_paths:
        tensor = image_to_mnist_tensor(image_path, device, show=show)

        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()

        results.append(f"Prediction for {os.path.basename(image_path)}: {pred}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Predict digit images using improved_digit_cnn.pth.")
    parser.add_argument("--model", type=str, default="improved_digit_cnn.pth", help="Path to model file")
    parser.add_argument(
        "--images",
        nargs="*",
        default=["datasets/digits/digit2.jpg", 
                "datasets/digits/digit4.jpg", 
                "datasets/digits/digit6.jpg", 
                "datasets/digits/digit8.jpg"
                ],
        help="Image files to predict",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument("--show", action="store_true", help="Show preprocessed images for debugging")
    args = parser.parse_args()

    lines = predict_images(args.model, args.images, args.device, show=args.show)
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()