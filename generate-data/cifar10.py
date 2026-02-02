import torch
from torchvision import datasets, transforms, utils
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

def save_image(img, path):
    utils.save_image(img, path)

def download_cifar10_subset(output_dir='./data/cifar10', per_class=100, max_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

    class_counts = defaultdict(int)
    tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, (img, label) in enumerate(dataset):
            if class_counts[label] < per_class:
                img_path = os.path.join(output_dir, f'{label}_{class_counts[label]}.jpg')
                tasks.append(executor.submit(save_image, img, img_path))
                class_counts[label] += 1

            if sum(class_counts.values()) >= per_class * 10:
                break

        # Wait for all tasks to finish
        for t in tasks:
            t.result()

    print(f"Saved {sum(class_counts.values())} CIFAR-10 images ({per_class} per class) in {output_dir}.")

download_cifar10_subset(per_class=500, max_workers=8)
