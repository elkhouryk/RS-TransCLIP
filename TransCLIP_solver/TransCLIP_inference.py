import argparse
import torch
import os
import csv
from .TransCLIP import TransCLIP_solver

def main(dataset_name, model_name, model_architecture, text_prompt, averageprompt):
    # Sanitize the text prompt
    sanitized_prompt = sanitize_prompt(text_prompt)

    # Setup paths
    features_path = f"./results/{dataset_name}/{model_name}/{model_architecture}"

    # Load features
    try:
        test_features = torch.load(os.path.join(features_path, "images.pt"))
        test_features /= test_features.norm(dim=-1, keepdim=True)
    except FileNotFoundError:
        print(f"File not found: {os.path.join(features_path, 'images.pt')}. Skipping.")
        return

    try:
        clip_prototypes = torch.load(os.path.join(features_path, f"texts_{sanitized_prompt}.pt"))
        clip_prototypes /= clip_prototypes.norm(dim=-1, keepdim=True)
        clip_prototypes = clip_prototypes.T
    except FileNotFoundError:
        print(f"File not found: {os.path.join(features_path, f'texts_{sanitized_prompt}.pt')}. Skipping.")
        return

    try:
        test_labels = torch.load(os.path.join(features_path, "classes.pt"))
    except FileNotFoundError:
        print(f"File not found: {os.path.join(features_path, 'classes.pt')}. Skipping.")
        return

    shot_features, shot_labels, val_features, val_labels = None, None, None, None

    # Run Solver
    _, _, acc_base_zs, acc_base_zs_trans = TransCLIP_solver(shot_features, shot_labels, val_features, val_labels, test_features, test_labels, clip_prototypes)
    
    # Save results to CSV
    save_results_to_csv(dataset_name, model_name, model_architecture, text_prompt, acc_base_zs, acc_base_zs_trans, averageprompt)

def sanitize_prompt(text_prompt):
    # Replace spaces with underscores
    return text_prompt.replace(" ", "_")

def save_results_to_csv(dataset_name, model_name, model_architecture, text_prompt, acc_base_zs, acc_base_zs_trans, averageprompt):
    if averageprompt == False:
        # Define the CSV file path
        csv_file_path = "./results/results_individualprompt.csv"
        
        # Check if the file already exists
        file_exists = os.path.isfile(csv_file_path)
        
        # Open the CSV file in append mode
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # If the file does not exist, write the header
            if not file_exists:
                writer.writerow(["dataset_name", "model_name", "model_architecture", "text_prompt", "acc_base_zs", "acc_base_zs_trans"])
            
            # Write the data
            writer.writerow([dataset_name, model_name, model_architecture, text_prompt, acc_base_zs, acc_base_zs_trans])
   
    elif averageprompt == True:
        # Define the CSV file path
        csv_file_path = "./results/results_averageprompt.csv"
        
        # Check if the file already exists
        file_exists = os.path.isfile(csv_file_path)
        
        # Open the CSV file in append mode
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # If the file does not exist, write the header
            if not file_exists:
                writer.writerow(["dataset_name", "model_name", "model_architecture", "text_prompt", "acc_base_averageprompt_zs", "acc_base_averageprompt_zs_trans"])
            
            # Write the data
            writer.writerow([dataset_name, model_name, model_architecture, text_prompt, acc_base_zs, acc_base_zs_trans])

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="TransCLIP Solver")
    parser.add_argument('--dataset_name', required=True, type=str, help='Name of the dataset')
    parser.add_argument('--model_name', required=True, type=str, help='Name of the model')
    parser.add_argument('--model_architecture', required=True, type=str, help='Architecture of the model')
    parser.add_argument('--text_prompt', required=True, type=str, help='Prompt to be sanitized')
    parser.add_argument('--averageprompt', required=False, type=str, help='Set to true when evaluating on average prompt')
    args = parser.parse_args()
    
    main(args.dataset_name, args.model_name, args.model_architecture, args.text_prompt, args.averageprompt)