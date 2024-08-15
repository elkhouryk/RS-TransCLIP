import os
import re
import torch
from PIL import Image
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import hf_hub_download
import open_clip
import clip
import zipfile
import requests
from io import BytesIO
import time
from transformers import AutoModelForCausalLM


class ImageDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = preprocess_image(image_path, self.preprocess)
        return image, image_path



def main(results_dir, datasets_dir, dataset_name, model_name, model_architecture, image_batch_size, image_fg, text_fg, gpu_id, num_workers, text_prompt):
    # Determine the device to use
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    '''
        Available dataset and model argument values:
        
        dataset_name:
            - AID
            - EuroSAT
            - fmow
            - MillionAID
            - PatternNet
            - RESISC45
            - RSICB256
            - MLRSNet
            - OPTIMAL31
            - RSC11
            - RSICB128
            - WHURS19
        
        model_name:
            - CLIP
            - RemoteCLIP
            - SkyCLIP30
            - SkyCLIP50
            - CLIP-LAION-RS
            - GeoRSCLIP
            
        model_architecture:
            - for CLIP: RN50, ViT-B-32, ViT-L-14, ViT-L-14-336
            - for RemoteCLIP: RN50, ViT-B-32, ViT-L-14
            - for SkyCLIP30: ViT-L-14
            - for SkyCLIP50: ViT-B-32, ViT-L-14
            - for CLIP-LAION-RS: ViT-L-14
            - for GeoRSCLIP: ViT-B-32, ViT-L-14, ViT-L-14-336, ViT-H-14
    '''
    


    # Directory containing the images
    image_directory = os.path.join("datasets", dataset_name, "images")
    
    # Check if the image directory exists
    if not os.path.exists(image_directory):
        raise FileNotFoundError(f"Image directory '{image_directory}' not found.")
    
    # Path to the classes.txt file
    classes_file = os.path.join("datasets", dataset_name, "classes.txt")
    # Path to the class_changes.txt file
    class_changes_file = os.path.join("datasets", dataset_name, "class_changes.txt")

    if model_name == 'CLIP':
        if re.match("ViT-L-14-336", model_architecture):
            temp_model_architecture= "ViT-L/14@336px"
            with tqdm(desc="Loading CLIP model", unit="model") as progress:
              model, preprocess = clip.load(temp_model_architecture, device=device)
              progress.update()
        elif re.match(r"ViT-(B|L)-\d+", model_architecture):  
            parts = model_architecture.split('-')     
            temp_model_architecture = f"{parts[0]}-{parts[1]}/{parts[2]}"            
            with tqdm(desc="Loading CLIP model", unit="model") as progress:
              model, preprocess = clip.load(temp_model_architecture, device=device)
              progress.update()         
        elif re.match("RN50", model_architecture):
            with tqdm(desc="Loading CLIP model", unit="model") as progress:
              model, preprocess = clip.load(model_architecture, device=device)
              progress.update() 
        else:
            raise ValueError("Invalid model architecture", model_architecture)           
            
    elif model_name == 'RemoteCLIP':
        if re.match(r"ViT-(B|L)-\d+", model_architecture):
            # Load the RemoteCLIP model and preprocessing function
            model, _, preprocess = open_clip.create_model_and_transforms(model_architecture)
            tokenizer = open_clip.get_tokenizer(model_architecture)
            # Download the pretrained checkpoint
            checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_architecture}.pt", cache_dir='RemoteCLIP_checkpoints')
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            message = model.load_state_dict(ckpt)
            print(message)
            model = model.to(device).eval()
        elif re.match("RN50", model_architecture):
            # Load the RemoteCLIP model and preprocessing function
            model, _, preprocess = open_clip.create_model_and_transforms(model_architecture)
            tokenizer = open_clip.get_tokenizer(model_architecture)
            # Download the pretrained checkpoint
            checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_architecture}.pt", cache_dir='RemoteCLIP_checkpoints')
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            message = model.load_state_dict(ckpt)
            print(message)
            model = model.to(device).eval() 
        else:
            raise ValueError("Invalid model architecture", model_architecture)

    elif model_name == 'SkyCLIP30':
        if model_architecture == "ViT-L-14":
            # Load the SkyCLIP30 model and preprocessing function
            model, _, preprocess = open_clip.create_model_and_transforms(model_architecture)
            tokenizer = open_clip.get_tokenizer(model_architecture)
            # Check if the pretrained checkpoint already exists
            checkpoint_path = "SkyCLIP_checkpoints/SkyCLIP_ViT_L14_top30pct/epoch_20.pt"
            if not os.path.exists(checkpoint_path):
                # Download and extract the pretrained checkpoint for SkyCLIP30
                zip_url = "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_L14_top30pct.zip"
                download_and_extract_zip(zip_url, "SkyCLIP_checkpoints")
            
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            
            # Extract the model state dictionary from the checkpoint
            model_state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            # Remove unexpected keys
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items() if k.startswith('module.')} 
            message = model.load_state_dict(model_state_dict, strict=False)
            print(message)
            model = model.to(device).eval()
        else:
            raise ValueError("Invalid model architecture", model_architecture)

    elif model_name == 'SkyCLIP50':
        if model_architecture == "ViT-L-14":
            # Load the SkyCLIP50 model and preprocessing function
            model, _, preprocess = open_clip.create_model_and_transforms(model_architecture)
            tokenizer = open_clip.get_tokenizer(model_architecture)
            # Check if the pretrained checkpoint already exists
            checkpoint_path = "SkyCLIP_checkpoints/SkyCLIP_ViT_L14_top50pct/epoch_20.pt"
            if not os.path.exists(checkpoint_path):
                # Download and extract the pretrained checkpoint for SkyCLIP50
                zip_url = "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_L14_top50pct.zip"
                download_and_extract_zip(zip_url, "SkyCLIP_checkpoints")
            
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            
            # Extract the model state dictionary from the checkpoint
            model_state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            # Remove unexpected keys
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items() if k.startswith('module.')} 
            message = model.load_state_dict(model_state_dict, strict=False)
            print(message)
            model = model.to(device).eval()

        elif model_architecture == "ViT-B-32":
            # Load the SkyCLIP50 model and preprocessing function
            model, _, preprocess = open_clip.create_model_and_transforms(model_architecture)
            tokenizer = open_clip.get_tokenizer(model_architecture)
            # Check if the pretrained checkpoint already exists
            checkpoint_path = "SkyCLIP_checkpoints/SkyCLIP_ViT_B32_top50pct/epoch_20.pt"
            if not os.path.exists(checkpoint_path):
                # Download and extract the pretrained checkpoint for SkyCLIP50
                zip_url = "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/SkyCLIP_ViT_B32_top50pct.zip"
                download_and_extract_zip(zip_url, "SkyCLIP_checkpoints")
            
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            
            # Extract the model state dictionary from the checkpoint
            model_state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            # Remove unexpected keys
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items() if k.startswith('module.')} 
            message = model.load_state_dict(model_state_dict, strict=False)
            print(message)
            model = model.to(device).eval()
        else:
            raise ValueError("Invalid model architecture", model_architecture)

    elif model_name == 'CLIP-LAION-RS':
        if model_architecture == "ViT-L-14":
            # Load the CLIP-LAION-RS model and preprocessing function
            model, _, preprocess = open_clip.create_model_and_transforms(model_architecture)
            tokenizer = open_clip.get_tokenizer(model_architecture)
            # Check if the pretrained checkpoint already exists
            checkpoint_path = "CLIP-LAION-RS_checkpoints/CLIP_ViT_L14_LAION_RS/epoch_20.pt"
            if not os.path.exists(checkpoint_path):
                # Download and extract the pretrained checkpoint for SkyCLIP50
                zip_url = "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/CLIP_ViT_L14_LAION_RS.zip"
                download_and_extract_zip(zip_url, "CLIP-LAION-RS_checkpoints")
            
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            
            # Extract the model state dictionary from the checkpoint
            model_state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            # Remove unexpected keys
            model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items() if k.startswith('module.')} 
            message = model.load_state_dict(model_state_dict, strict=False)
            print(message)
            model = model.to(device).eval()
        else:
            raise ValueError("Invalid model architecture", model_architecture)
 
    elif model_name == 'GeoRSCLIP':
        if re.match("ViT-(B|L|H)-\d+-\d+", model_architecture):
            # Load the GeoRSCLIP model and preprocessing function
            model, _, preprocess = open_clip.create_model_and_transforms(model_architecture)
            tokenizer = open_clip.get_tokenizer(model_architecture)
            # Download the pretrained checkpoint
            checkpoint_path = hf_hub_download("Zilun/GeoRSCLIP", f"ckpt/RS5M_{model_architecture}.pt", cache_dir='GeoRSCLIP_checkpoints')
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            message = model.load_state_dict(ckpt)
            print(message)
            model = model.to(device).eval()
        elif re.match(r"ViT-(B|L|H)-\d+", model_architecture):
            # Load the GeoRSCLIP model and preprocessing function
            model, _, preprocess = open_clip.create_model_and_transforms(model_architecture)
            tokenizer = open_clip.get_tokenizer(model_architecture)
            # Download the pretrained checkpoint
            checkpoint_path = hf_hub_download("Zilun/GeoRSCLIP", f"ckpt/RS5M_{model_architecture}.pt", cache_dir='GeoRSCLIP_checkpoints')
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            message = model.load_state_dict(ckpt)
            print(message)
            model = model.to(device).eval()

        else:
            raise ValueError("Invalid model architecture", model_architecture)
 
 
            
    else:
      raise ValueError("Invalid model name", model_name)



    # Get list of image paths
    image_paths = get_image_paths(image_directory)
    # Get list of class labels
    class_labels = get_class_labels(classes_file)
    # Get new class names from the class_changes file
    new_class_names = get_class_changes(class_changes_file)

    # Create text prompts for each class
    text_prompts = [f"{text_prompt} {new_class_names[i]}." for i in range(len(class_labels))]

    if model_name == 'CLIP':   
        # Tokenize text prompts
        text = clip.tokenize(text_prompts).to(device)
    elif model_name in ['RemoteCLIP', 'SkyCLIP30', 'SkyCLIP50', "CLIP-LAION-RS", "GeoRSCLIP"]:
        # Tokenize text prompts
        text = tokenizer(text_prompts).to(device)     

    image_features_list = []
    text_features_list = []
    image_classes_list = []
    
    # Set batch size and create data loader
    batch_size = image_batch_size
    dataset = ImageDataset(image_paths, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    if image_fg:       
        # Process images in batches with a progress bar
        print("Generating image features...")
        
        start_time = time.time()  # Start timing

        
        for images, batch_paths in tqdm(dataloader, desc="Processing Images"):
            try:
                images = images.to(device)

                with torch.no_grad():
                    # Encode the batch of images to get their features
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features_list.append(image_features)

                # Extract class indices from image filenames
                batch_classes = [extract_class_from_filename(path, class_labels) for path in batch_paths]
                image_classes_list.extend(batch_classes)
            except Exception as e:
                print(f"Error processing batch: {e}")
    
    
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        print(f"Total time taken to process images: {elapsed_time:.2f} seconds")
    
    if text_fg:   
        # Process each text prompt with a progress bar
        print("Generating text features...")
        
        start_time = time.time()  # Start timing
        
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
    
        # Add text features to list
        for i in tqdm(range(len(class_labels)), desc="Processing Texts"):
            text_features_list.append(text_features[i])
            
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        print(f"Total time taken to process texts: {elapsed_time:.2f} seconds")
    
    # Combine all image features into a single tensor
    if image_features_list:
        all_image_features = torch.cat(image_features_list, dim=0)
        # Construct save path
        save_path = os.path.join("results", dataset_name, model_name, model_architecture, 'images.pt')
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the combined image features
        torch.save(all_image_features, save_path)

    # Combine all text features into a single tensor
    if text_features_list:
        all_text_features = torch.stack(text_features_list, dim=0)
        # Construct save path
        sanitized_prompt = sanitize_prompt(text_prompt)
        save_path = os.path.join("results", dataset_name, model_name, model_architecture, f"texts_{sanitized_prompt}.pt")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the combined image features
        torch.save(all_text_features, save_path)

    if image_fg: 
        # Save the class labels for each image
        save_path = os.path.join("results", dataset_name, model_name, model_architecture, 'classes.pt')
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(torch.tensor(image_classes_list), save_path)

    print("All features and class labels saved successfully.")

def sanitize_prompt(text_prompt):
    # Replace spaces with underscores
    return text_prompt.replace(" ", "_")

def get_image_paths(directory):
    """
    Get a list of all files in the specified directory.
    """
    file_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
    if not file_paths:
        raise FileNotFoundError(f"No files found in directory '{directory}'.")
    return file_paths

def get_class_labels(filepath):
    """
    Get a list of class labels from a text file, where each line is a class.
    """
    try:
        with open(filepath, 'r') as file:
            class_labels = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Class file '{filepath}' not found.")
    if not class_labels:
        raise ValueError(f"No class labels found in file '{filepath}'.")
    return class_labels

def get_class_changes(filepath):
    """
    Get a list of new class names from a text file, where each line is a new class name.
    """
    try:
        with open(filepath, 'r') as file:
            class_changes = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Class changes file '{filepath}' not found.")
    if not class_changes:
        raise ValueError(f"No class changes found in file '{filepath}'.")
    return class_changes

def preprocess_image(image_path, preprocess):
    """
    Preprocess an image given its path and returns the preprocessed image tensor.
    """
    # Disable the decompression bomb warning
    Image.MAX_IMAGE_PIXELS = None
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise Exception(f"Error opening image file '{image_path}': {e}")
    
    image = preprocess(image)
    return image

def extract_class_from_filename(filename, class_labels):
    """
    Extract the class index from the image filename.
    Assumes the filename format is '{class}_id.jpg'.
    """
    basename = os.path.basename(filename)
    class_name = basename.split('_')[0]
    if class_name in class_labels:
        return class_labels.index(class_name)
    else:
        raise ValueError(f"Class '{class_name}' not found in class labels.")


def download_and_extract_zip(url, extract_to='.', retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                break
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    else:
        raise Exception("Failed to download file after multiple attempts.")
    
    # Check if the response is successful
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {response.status_code}")
    
    # Print the content type to verify it is a zip file
    content_type = response.headers.get('Content-Type', '')
    print(f"Content-Type: {content_type}")
    
    if 'zip' not in content_type:
        raise Exception(f"Unexpected content type: {content_type}")
    
    # Get the total length of the content
    total_length = int(response.headers.get('Content-Length', 0))
    
    # Download the file with a progress bar
    downloaded_content = BytesIO()
    with tqdm(total=total_length, unit='B', unit_scale=True, desc='Downloading') as pbar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                downloaded_content.write(chunk)
                pbar.update(len(chunk))
    
    downloaded_content.seek(0)  # Go to the beginning of the BytesIO object

    # Print the size of the downloaded content
    print(f"Downloaded file size: {downloaded_content.getbuffer().nbytes} bytes")
    
    # Save the response content to a file for manual inspection (optional)
    with open('downloaded_file.zip', 'wb') as f:
        f.write(downloaded_content.getbuffer())
    
    # Check if the content is indeed a zip file
    try:
        with zipfile.ZipFile(downloaded_content) as zip_ref:
            zip_ref.extractall(extract_to)
    except zipfile.BadZipFile as e:
        print("Failed to extract zip file. The file might be corrupted.")
        raise e    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature generation on satellite images using various models.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--datasets_dir", type=str, required=True, help="Directory containing datasets.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model.")
    parser.add_argument("--model_architecture", type=str, required=True, help="Model architecture.")
    parser.add_argument("--image_batch_size", type=int, default=1, help="Batch size for image processing.")
    parser.add_argument("--image_fg", action="store_true", help="Flag to run image feature generation.")
    parser.add_argument("--text_fg", action="store_true", help="Flag to run text feature generation.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--text_prompt", type=str, default="a satellite photo of a", help="Text prompt for generating class descriptions.")
    
    args = parser.parse_args()
    
    main(args.results_dir, args.datasets_dir, args.dataset_name, args.model_name, args.model_architecture, args.image_batch_size, args.image_fg, args.text_fg, args.gpu_id, args.num_workers, args.text_prompt)