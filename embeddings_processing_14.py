import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Constants
OUTPUT_CSV = "image_embeddings_mil_15"
IMG_SIZE = (224, 224)  # Image size for Ludwig
EMBEDDING_SIZE = 1000  # Fixed size for all embeddings
BAG_SIZE = 100  # Number of images per bag

def load_dataset(dataset_type):
    if dataset_type == "all_data":
        return pd.read_csv("./../er_status_all_data.csv")
    if dataset_type == "no_white":
        return pd.read_csv("./../er_status_no_white.csv")
    return None

def get_unique_samples(load_data):
    np.random.seed(42)
    unique_samples = load_data["sample"].unique()
    num_samples = len(unique_samples)
    print(f"Unique samples count: {num_samples}", flush=True)
    return unique_samples, num_samples

def split_data(unique_samples, num_samples, data):
    train_count = int(0.7 * num_samples)
    val_count = int(0.1 * num_samples)
    shuffled_samples = np.random.permutation(unique_samples)
    train_samples = shuffled_samples[:train_count]
    val_samples = shuffled_samples[train_count:train_count + val_count]
    test_samples = shuffled_samples[train_count + val_count:]

    sample_to_split = {sample: 0 for sample in train_samples}
    sample_to_split.update({sample: 1 for sample in val_samples})
    sample_to_split.update({sample: 2 for sample in test_samples})

    data["split"] = data["sample"].map(sample_to_split)
    return data, train_samples, val_samples, test_samples

def init_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def extract_resnet_embedding(image_path, transform, resnet):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(image).squeeze().cpu().numpy()

    if embedding.size != EMBEDDING_SIZE:
        embedding = np.pad(embedding, (0, EMBEDDING_SIZE - embedding.size), 'constant') if embedding.size < EMBEDDING_SIZE else embedding[:EMBEDDING_SIZE]
    return embedding

def attention_pooling(embeddings):
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    attention_weights = F.softmax(embeddings_tensor @ embeddings_tensor.T, dim=1)  # Self-attention
    pooled_embedding = attention_weights @ embeddings_tensor
    return pooled_embedding.mean(dim=0).numpy()

def convert_embedding_to_string(embedding):
    embedding = np.nan_to_num(embedding)
    return " ".join(map(str, embedding))

def create_bags(split_data, split, transform, resnet):
    print(f"Creating bags for split {split}", flush=True)
    bags = []
    sample_groups = split_data.groupby("sample")

    for sample, group in sample_groups:
        embeddings = []
        labels = []
        image_paths = group["image_path"].values
        num_images = len(image_paths)

        for idx, image_path in enumerate(image_paths):
            embedding = extract_resnet_embedding(image_path, transform, resnet)
            embeddings.append(embedding)
            labels.append(group.iloc[idx]["er_status_by_ihc"])
            
            if len(embeddings) == BAG_SIZE or (idx == num_images - 1 and len(embeddings) > 0):
                aggregated_embedding = attention_pooling(np.array(embeddings))
                embedding_string = convert_embedding_to_string(aggregated_embedding)
                bag_label = int(any(np.array(labels) == 1))
                bags.append({
                    "embedding": embedding_string,
                    "bag_label": bag_label,
                    "split": split,
                    "bag_size": len(embeddings),
                    "sample": sample
                })
                embeddings = []
                labels = []
    return bags

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=False).to(device)
weights_path = "./../resnet50-0676ba61.pth"
resnet.load_state_dict(torch.load(weights_path, map_location=device))
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

for dataset in ["all_data", "no_white"]:
    data = load_dataset(dataset)
    unique_samples, num_samples = get_unique_samples(data)
    data, train_samples, val_samples, test_samples = split_data(unique_samples, num_samples, data)
    transform = init_transform()

    all_bags = []
    for split, split_samples in [(0, train_samples), (1, val_samples), (2, test_samples)]:
        split_data_subset = data[data["sample"].isin(split_samples)]
        split_bags = create_bags(split_data_subset, split, transform, resnet)
        all_bags.extend(split_bags)

    metadata_df = pd.DataFrame(all_bags)
    output_csv = f"{OUTPUT_CSV}_{dataset}.csv"
    metadata_df.to_csv(output_csv, index=False)
    print(f"Saved metadata to {output_csv}")
