#!/usr/bin/env python3
# text_extraction.py

import argparse
import os
import re
import csv
from pathlib import Path
import cv2
import pytesseract

PRICE_PATTERN = re.compile(r'\$?\d+(?:\.\d{2})')

def list_images(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    return sorted(
        str(p) for p in Path(folder).rglob('*')
        if p.suffix.lower() in exts
    )

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    scale = 2.0
    gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return thresh

def extract_store_name(text_lines):
    for line in text_lines[:5]:
        clean = line.strip()
        if clean and any(c.isalpha() for c in clean):
            return clean
    return "Unknown Store"

def extract_items_and_prices(text_lines):
    items = []

    for line in text_lines:
        line = line.strip()
        if not line:
            continue

        matches = PRICE_PATTERN.findall(line)
        if not matches:
            continue

        price_text = matches[-1].replace('$', '')
        try:
            price = float(price_text)
        except ValueError:
            continue

        item_name = line.replace(matches[-1], '').strip(" .:-")
        if not item_name:
            item_name = "Unknown Item"

        items.append((item_name, price))

    return items

def process_receipt(image_path):
    processed = preprocess_image(image_path)

    text = pytesseract.image_to_string(processed, config='--psm 6')
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    store = extract_store_name(lines)
    items = extract_items_and_prices(lines)

    total = sum(price for _, price in items)

    return store, items, total

def write_csv(rows, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['store', 'item', 'amount'])
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description="Extract receipt text into CSV using Tesseract OCR.")
    parser.add_argument('--folder', type=str, default='datasets/receipts', help='Folder containing receipt images')
    parser.add_argument('--out', type=str, default='shopping_summary.csv', help='Output CSV file')
    args = parser.parse_args()

    image_files = list_images(args.folder)
    if not image_files:
        raise SystemExit("No receipt images found.")

    rows = []

    for image_path in image_files:
        try:
            store, items, total = process_receipt(image_path)

            for item_name, price in items:
                rows.append([store, item_name, f"{price:.2f}"])

            rows.append([store, 'Total', f"{total:.2f}"])

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    write_csv(rows, args.out)
    print(f"CSV written to {args.out}")

if __name__ == '__main__':
    main()
