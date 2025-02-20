import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

# Constants
OUTPUT_CSV = "image_embeddings_mil_17"
IMG_SIZE = (224, 224)  # Image size for model input
EMBEDDING_SIZE = 1000  # Fixed size for embeddings
MIN_BAG_SIZE = 100  # Minimum bag size for random sampling
MAX_BAG_SIZE = 150  # Maximum bag size for random sampling
NUM_ITERATIONS = 10  # Number of iterations for train/val bags

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

def create_bags(split_data, split, transform, model, device, iteration=None):
    print(f"Creating bags for split {split}" + (f", iteration {iteration}" if iteration is not None else ""), flush=True)
    bags = []
    
    # For training (0) and validation (1) splits - random bags across samples
    if split in [0, 1]:
        # Get all embeddings first
        embeddings_list = []
        labels_list = []
        sample_ids = []
        
        for _, row in split_data.iterrows():
            embedding = extract_resnet_embedding(row["image_path"], transform, model, device)
            embeddings_list.append(embedding)
            labels_list.append(row["er_status_by_ihc"])
            sample_ids.append(row["sample"])
            
        embeddings_array = np.array(embeddings_list)
        labels_array = np.array(labels_list)
        
        # Create random bags
        indices = np.arange(len(embeddings_array))
        np.random.shuffle(indices)
        current_idx = 0
        
        while current_idx < len(indices):
            bag_size = np.random.randint(MIN_BAG_SIZE, MAX_BAG_SIZE + 1)
            bag_indices = indices[current_idx:current_idx + bag_size]
            if len(bag_indices) < MIN_BAG_SIZE//2:  # Avoid very small leftover bags
                break
                
            bag_embeddings = embeddings_array[bag_indices]
            bag_labels = labels_array[bag_indices]
            bag_samples = [sample_ids[i] for i in bag_indices]
            
            aggregated_embedding = aggregate_embeddings(bag_embeddings)
            bag_label = int(any(bag_labels == 1))
            
            bags.append({
                "embedding": " ".join(map(str, aggregated_embedding)),
                "bag_label": bag_label,
                "split": split,
                "bag_size": len(bag_indices),
                "samples": " ".join(map(str, bag_samples)),  # Store all sample IDs in the bag
                "iteration": iteration if iteration is not None else 0
            })
            current_idx += bag_size
            
    # For testing (2) split - keep original behavior (group by sample)
    else:
        for sample, group in split_data.groupby("sample"):
            embeddings, labels = [], []
            
            for _, row in group.iterrows():
                embedding = extract_resnet_embedding(row["image_path"], transform, model, device)
                embeddings.append(embedding)
                labels.append(row["er_status_by_ihc"])
                
            aggregated_embedding = aggregate_embeddings(np.array(embeddings))
            bag_label = int(any(np.array(labels) == 1))
            bags.append({
                "embedding": " ".join(map(str, aggregated_embedding)),
                "bag_label": bag_label,
                "split": split,
                "bag_size": len(embeddings),
                "samples": str(sample),
                "iteration": 0
            })

    return bags

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=False).to(device)
resnet.load_state_dict(torch.load("./../resnet50-0676ba61.pth", map_location=device))
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# Process each dataset
for dataset in ["all_data", "no_white"]:
    data = load_dataset(dataset)
    unique_samples, _ = get_unique_samples(data)
    data, train_samples, val_samples, test_samples = split_data(unique_samples, data)
    transform = init_transform()

    all_bags = []
    
    # Create 10 iterations of random bags for train and validation
    for iteration in range(NUM_ITERATIONS):
        for split, split_samples in [(0, train_samples), (1, val_samples)]:
            split_data_subset = data[data["sample"].isin(split_samples)]
            all_bags.extend(create_bags(split_data_subset, split, transform, resnet, device, iteration))
    
    # Create bags for test split (single iteration)
    split_data_subset = data[data["sample"].isin(test_samples)]
    all_bags.extend(create_bags(split_data_subset, 2, transform, resnet, device))

    # Save results
    metadata_df = pd.DataFrame(all_bags)
    output_csv = f"{OUTPUT_CSV}_{dataset}.csv"
    metadata_df.to_csv(output_csv, index=False)
    print(f"Saved metadata to {output_csv}")
