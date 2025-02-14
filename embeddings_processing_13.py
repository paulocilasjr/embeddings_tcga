import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler



# Updated Paths and Constants
OUTPUT_CSV = "image_embeddings_mil_14"
IMG_SIZE = (224, 224)  # Image size for Ludwig
EMBEDDING_SIZE = 1000  # Fixed size for all embeddings, you can adjust this based on the model

# Load Data
def LoadDataset (dataset_type):
    if dataset_type == "all_data":
        dataset = pd.read_csv("./../er_status_all_data.csv")
    if dataset_type == "no_white":
        dataset = pd.read_csv("./../er_status_no_white.csv")
    return dataset

def UniqueSamples (load_data):
    # Assign Split Randomly at the Sample Level
    np.random.seed(42)  # Set a seed for reproducibility
    unique_samples = load_data["sample"].unique()
    num_samples = len(unique_samples)
    print(f"this is the length of unique samples: {num_samples}", flush=True)
    return unique_samples, num_samples

def SplitData (unique_samples, num_samples, data):
    # Generate random splits for samples
    train_count = int(0.7 * num_samples)
    val_count = int(0.1 * num_samples)

    # Randomly shuffle the samples
    shuffled_samples = np.random.permutation(unique_samples)

    # Assign splits
    train_samples = shuffled_samples[:train_count]
    val_samples = shuffled_samples[train_count:train_count + val_count]
    test_samples = shuffled_samples[train_count + val_count:]

    # Map samples to splits
    sample_to_split = {sample: 0 for sample in train_samples}  # Train: 0
    sample_to_split.update({sample: 1 for sample in val_samples})  # Validation: 1
    sample_to_split.update({sample: 2 for sample in test_samples})  # Test: 2

    # Add split column to the DataFrame
    data["split"] = data["sample"].map(sample_to_split)

    return data, train_samples, val_samples, test_samples 

def TransformInitiation():
    # Preprocessing Transformations
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform

# Function to Create ResNet Embeddings from Image
def extract_resnet_embedding(image_path, transform, resnet):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = resnet(image)

    # Remove the classification head (last layer) and use the features from earlier layers
    embedding = embedding.squeeze().cpu().numpy()  # Convert tensor to NumPy array

    # Ensure the embedding has a consistent size
    if embedding.size != EMBEDDING_SIZE:
        print(f"Warning: embedding size {embedding.size} is not {EMBEDDING_SIZE}. Padding or truncating.")
        # Pad or truncate the embedding to the fixed size
        embedding = np.pad(embedding, (0, EMBEDDING_SIZE - embedding.size), 'constant') if embedding.size < EMBEDDING_SIZE else embedding[:EMBEDDING_SIZE]

    return embedding

# Function to Aggregate Embeddings Using Max Pooling
def aggregate_embeddings_with_max_pooling(embeddings):
    print("inside aggregate_embeddings_with_max_pooling function", flush=True)
    # Apply max pooling across the embeddings (along the first axis)
    max_pooled_embedding = np.max(embeddings, axis=0)
    return max_pooled_embedding

# Function to Save Aggregated Embedding as String
def convert_embedding_to_string(embedding):
    embedding = np.nan_to_num(embedding)  # Replace NaNs with zero
    return " ".join(map(str, embedding))

# Function to Create Bags from Split Data
def create_bags_from_split(split_data, split, repeats, transform, resnet):
    print("inside create bags from split function", flush=True)
    bags = []
    images = []
    bag_set = set()

    # Collect all images for the current split
    for _, row in split_data.iterrows():
        image_path = row["image_path"]
        embedding = extract_resnet_embedding(image_path, transform, resnet)  # Pass transform and resnet
        images.append((embedding, row["er_status_by_ihc"], row["sample"]))

    for _ in range(repeats):
        images_0 = [image for image in images if image[1] == 0]
        images_1 = [image for image in images if image[1] == 1]

        # Ensure randomness by shuffling both groups
        np.random.shuffle(images_0)
        np.random.shuffle(images_1)

        make_bag_1 = True

        # Continue until all images are used
        while len(images_0) + len(images_1) > 0:
            # Determine random bag size between 3 and 7
            #bag_size = np.random.randint(3, 8)
            bag_size = 40
            if make_bag_1 and len(images_1) > 0:
                print("TURN = 1")
                if len(images_0) > 0:
                    num_1_tiles = np.random.randint(1, bag_size + 1)
                else:
                    num_1_tiles = bag_size

                selected_images_1 = images_1[:num_1_tiles]
                images_1 = images_1[num_1_tiles:]  # Remove selected images

                # Fill the rest of the bag with images_0
                num_0_tiles = min(bag_size - num_1_tiles, len(images_0))
                selected_images_0 = images_0[:num_0_tiles]
                images_0 = images_0[num_0_tiles:]

                make_bag_1 = False
                # Combine selected images to form the bag
                bag_images = selected_images_1 + selected_images_0
                if len(bag_images) != bag_size:
                    num_extra_tiles = bag_size - len(bag_images)
                    selected_images_extra = images_1[:num_extra_tiles]
                    images_1 = images_1[num_extra_tiles:]

                    # Combine selected images to form the bag
                    bag_images += selected_images_extra

            elif not make_bag_1 and len(images_0) > 0:
                print("TURN = 0")
                num_0_tiles = bag_size
                selected_images_0 = images_0[:num_0_tiles]
                print(f"Num of tiles 0 taken: {num_0_tiles}")
                images_0 = images_0[num_0_tiles:]
                selected_images_1 = []
                make_bag_1 = True
                bag_images = selected_images_0

            else:
                print("it is going to pass")
                make_bag_1 = not make_bag_1
                bag_images = []

            if len(bag_images) > 0:
                # Extract data for bag-level representation
                bag_image_embeddings = [x[0] for x in bag_images]
                bag_labels = [x[1] for x in bag_images]
                bag_samples = [x[2] for x in bag_images]

                # Aggregate images into a single embedding using max pooling
                aggregated_embedding = aggregate_embeddings_with_max_pooling(np.array(bag_image_embeddings))

                # Convert the aggregated embedding to a string for the DataFrame
                embedding_string = convert_embedding_to_string(aggregated_embedding)

                # Bag-level label (if any image has label 1, the bag label is 1)
                bag_label = int(any(np.array(bag_labels) == 1))

                bag_image_embeddings_tuple = tuple(map(tuple, bag_image_embeddings))
                bag_samples_tuple = tuple(bag_samples)
                bag_key = (bag_image_embeddings_tuple, len(bag_images), bag_samples_tuple)

                if bag_key not in bag_set:
                    bag_set.add(bag_key)

                    # Add the bag information to the records
                    bags.append({
                        "embedding": embedding_string,
                        "bag_label": bag_label,
                        "split": split,
                        "bag_size": len(bag_images),
                        "bag_samples": bag_samples
                    })
                else:
                    print("A bag was created twice", flush = True)
    return bags

# ResNet Model Setup for Embedding Extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=False).to(device)
weights_path = "./../resnet50-0676ba61.pth"
resnet.load_state_dict(torch.load(weights_path, map_location=device))
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval() 

# Create Bags for All Splits
for dataset in ["all_data", "no_white"]:
    data = LoadDataset(dataset)
    unique_samples, num_samples = UniqueSamples(data)
    data, train_samples, val_samples, test_samples = SplitData (unique_samples, num_samples, data)
    transform = TransformInitiation()

    all_bags = []
    for split, split_samples in [(0, train_samples), (1, val_samples), (2, test_samples)]:
        print(f"create bags for: {split}", flush=True)
        split_data = data[data["sample"].isin(split_samples)]
        split_bags = create_bags_from_split(split_data, split, 1, transform, resnet)
        all_bags.extend(split_bags)

    # Save Metadata to CSV
    metadata_df = pd.DataFrame(all_bags)
    output_csv = f"{OUTPUT_CSV}_{dataset}.csv"
    metadata_df.to_csv(output_csv, index=False)
    print(f"Saved balanced metadata to {output_csv}")
