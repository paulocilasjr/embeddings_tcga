import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

# Constants
OUTPUT_CSV = "image_embeddings_mil_16"
IMG_SIZE = (224, 224)  # Image size for model input
EMBEDDING_SIZE = 1000  # Fixed size for embeddings
BAG_SIZE = 100  # Fixed bag size per sample

def load_dataset(dataset_type):
    file_map = {
        "all_data": "./../er_status_all_data.csv",
        "no_white": "./../er_status_no_white.csv"
    }
    return pd.read_csv(file_map.get(dataset_type, "")) if dataset_type in file_map else None

def get_unique_samples(data):
    np.random.seed(42)
    unique_samples = data["sample"].unique()
    np.random.shuffle(unique_samples)
    return unique_samples, len(unique_samples)

def split_data(unique_samples, data):
    num_samples = len(unique_samples)
    train_count = int(0.7 * num_samples)
    val_count = int(0.1 * num_samples)
    
    train_samples = unique_samples[:train_count]
    val_samples = unique_samples[train_count:train_count + val_count]
    test_samples = unique_samples[train_count + val_count:]

    sample_to_split = {s: i for i, samples in enumerate([train_samples, val_samples, test_samples]) for s in samples}
    data["split"] = data["sample"].map(sample_to_split)
    return data, train_samples, val_samples, test_samples

def init_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_resnet_embedding(image_path, transform, model, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image).squeeze().cpu().numpy()
    return np.pad(embedding, (0, EMBEDDING_SIZE - embedding.size), 'constant') if embedding.size < EMBEDDING_SIZE else embedding[:EMBEDDING_SIZE]

def aggregate_embeddings(embeddings):
    return np.max(embeddings, axis=0)  # Max pooling

def create_bags(split_data, split, transform, model, device):
    print(f"Creating bags for split {split}", flush=True)
    bags = []

    for sample, group in split_data.groupby("sample"):
        embeddings, labels = [], []
        
        for _, row in group.iterrows():
            embedding = extract_resnet_embedding(row["image_path"], transform, model, device)
            embeddings.append(embedding)
            labels.append(row["er_status_by_ihc"])

            if len(embeddings) == BAG_SIZE:
                aggregated_embedding = aggregate_embeddings(np.array(embeddings))
                bag_label = int(any(np.array(labels) == 1))
                bags.append({
                    "embedding": " ".join(map(str, aggregated_embedding)),
                    "bag_label": bag_label,
                    "split": split,
                    "bag_size": BAG_SIZE,
                    "sample": sample
                })
                embeddings, labels = [], []  # Reset for next bag
        
        if embeddings:  # Process remaining images (if less than 100)
            aggregated_embedding = aggregate_embeddings(np.array(embeddings))
            bag_label = int(any(np.array(labels) == 1))
            bags.append({
                "embedding": " ".join(map(str, aggregated_embedding)),
                "bag_label": bag_label,
                "split": split,
                "bag_size": len(embeddings),
                "sample": sample
            })
    return bags

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=False).to(device)
resnet.load_state_dict(torch.load("./../resnet50-0676ba61.pth", map_location=device))
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

for dataset in ["all_data", "no_white"]:
    data = load_dataset(dataset)
    unique_samples, _ = get_unique_samples(data)
    data, train_samples, val_samples, test_samples = split_data(unique_samples, data)
    transform = init_transform()

    all_bags = []
    for split, split_samples in [(0, train_samples), (1, val_samples), (2, test_samples)]:
        split_data_subset = data[data["sample"].isin(split_samples)]
        all_bags.extend(create_bags(split_data_subset, split, transform, resnet, device))

    metadata_df = pd.DataFrame(all_bags)
    output_csv = f"{OUTPUT_CSV}_{dataset}.csv"
    metadata_df.to_csv(output_csv, index=False)
    print(f"Saved metadata to {output_csv}")
