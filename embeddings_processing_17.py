import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

# Constants
OUTPUT_CSV = "image_embeddings_mil_18"
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
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(image).squeeze().cpu().numpy()
        return np.pad(embedding, (0, EMBEDDING_SIZE - embedding.size), 'constant') if embedding.size < EMBEDDING_SIZE else embedding[:EMBEDDING_SIZE]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return np.zeros(EMBEDDING_SIZE)  # Return zero vector on failure

def aggregate_embeddings(embeddings):
    return np.max(embeddings, axis=0)  # Max pooling

def create_bags(split_data, split, transform, model, device, iteration=None):
    print(f"Creating bags for split {split}" + (f", iteration {iteration}" if iteration is not None else ""), flush=True)
    bags = []
    
    # For training (0) and validation (1) splits - use "turns" approach
    if split in [0, 1]:
        # Prepare data
        embeddings_0 = []
        embeddings_1 = []
        for _, row in split_data.iterrows():
            embedding = extract_resnet_embedding(row["image_path"], transform, model, device)
            sample_info = (row["sample"], row["er_status_by_ihc"], embedding)
            if row["er_status_by_ihc"] == 1:
                embeddings_1.append(sample_info)
            else:
                embeddings_0.append(sample_info)
        
        np.random.shuffle(embeddings_0)
        np.random.shuffle(embeddings_1)
        
        make_bag_1 = True  # Start with positive bag
        bag_set = set()    # Track unique bags
        
        while embeddings_0 or embeddings_1:
            bag_size = np.random.randint(MIN_BAG_SIZE, MAX_BAG_SIZE + 1)
            
            if make_bag_1 and embeddings_1:
                num_1_samples = min(np.random.randint(1, bag_size + 1), len(embeddings_1))
            else:
                num_1_samples = 0
            
            selected_embeddings_1 = embeddings_1[:num_1_samples]
            embeddings_1 = embeddings_1[num_1_samples:]
            
            num_0_samples = min(bag_size - num_1_samples, len(embeddings_0))
            selected_embeddings_0 = embeddings_0[:num_0_samples]
            embeddings_0 = embeddings_0[num_0_samples:]
            
            # Combine embeddings
            bag_data = selected_embeddings_0 + selected_embeddings_1
            if len(bag_data) < MIN_BAG_SIZE // 2:  # Avoid tiny leftover bags
                break
            
            sample_names = [x[0] for x in bag_data]
            sample_labels = [x[1] for x in bag_data]
            only_embeddings = [x[2] for x in bag_data]
            
            aggregated_embedding = aggregate_embeddings(np.array(only_embeddings))
            bag_label = int(any(np.array(sample_labels) == 1))
            
            # Create a unique key for the bag
            bag_key = (tuple(map(tuple, only_embeddings)), len(bag_data), tuple(sample_names))
            if bag_key not in bag_set:
                bag_set.add(bag_key)
                bags.append({
                    "embedding": " ".join(map(str, aggregated_embedding)),
                    "bag_label": bag_label,
                    "split": split,
                    "bag_size": len(bag_data),
                    "samples": " ".join(map(str, sample_names)),
                    "iteration": iteration if iteration is not None else 0
                })
            
            make_bag_1 = not make_bag_1  # Toggle for next bag
        
    # For testing (2) split - group by sample, single creation
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
                "iteration": 0  # Fixed iteration for test
            })

    return bags

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=False).to(device)
resnet.load_state_dict(torch.load("./../resnet50-0676ba61.pth", map_location=device))
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove classification layer
resnet.eval()

# Process each dataset
for dataset in ["all_data", "no_white"]:
    data = load_dataset(dataset)
    if data is None:
        print(f"Dataset {dataset} not found.")
        continue
    
    unique_samples, _ = get_unique_samples(data)
    data, train_samples, val_samples, test_samples = split_data(unique_samples, data)
    transform = init_transform()

    all_bags = []
    
    # Create 10 iterations of random bags for train and validation
    for iteration in range(NUM_ITERATIONS):
        for split, split_samples in [(0, train_samples), (1, val_samples)]:
            split_data_subset = data[data["sample"].isin(split_samples)]
            all_bags.extend(create_bags(split_data_subset, split, transform, resnet, device, iteration))
    
    # Create bags for test split (single creation, no iterations)
    split_data_subset = data[data["sample"].isin(test_samples)]
    all_bags.extend(create_bags(split_data_subset, 2, transform, resnet, device))

    # Save results
    metadata_df = pd.DataFrame(all_bags)
    output_csv = f"{OUTPUT_CSV}_{dataset}.csv"
    metadata_df.to_csv(output_csv, index=False)
    print(f"Saved metadata to {output_csv}")
