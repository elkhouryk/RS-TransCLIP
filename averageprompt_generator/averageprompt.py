import os
import torch
import glob

def main(dataset_name, model_name, model_architecture):
    base_path = f"./results/{dataset_name}/{model_name}/{model_architecture}"
    
    # Walk through all directories under the base path
    for root, dirs, files in os.walk(base_path):
        tensor_files = glob.glob(os.path.join(root, 'texts_*.pt'))
        
        if not tensor_files:
            continue
        
        tensors = []
        
        for file in tensor_files:
            tensor = torch.load(file)
            tensors.append(tensor)
        
        if tensors:
            # Stack tensors and calculate the mean along the first dimension
            stacked_tensors = torch.stack(tensors)
            mean_tensor = torch.mean(stacked_tensors, dim=0)
            mean_tensor /= mean_tensor.norm(dim=-1, keepdim=True)
            
            
            
            # Save the normalized mean tensor
            output_path = os.path.join(root, 'texts_averageprompt.pt')
            torch.save(mean_tensor, output_path)
            print(f"Saved normalized mean tensor to {output_path}")

