#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import argparse
import os
from datasets import load_dataset
import shutil 
import subprocess
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

#%%
def save_images_from_dataloader(load_path, output_dir, classes, label_counters={}, split='train', label_key='label'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    ds = load_dataset(load_path, split=split)
    # Set the format of the dataset to PyTorch tensors
    ds.set_format(type='torch', columns=['image', label_key])
    
    dataloader = DataLoader(ds, batch_size=1, shuffle=True)
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing dataset")):
        images = batch['image']
        labels = batch[label_key]
        for i in range(images.size(0)):
            label = str(labels[i].item())
            image = images[i]
            
            # Ensure the label has a counter
            if label not in label_counters:
                label_counters[label] = 0
            label_counters[label] += 1
            
            # Create filename
            c = classes[label].replace(' ', '')
            filename = f"{c}_{label_counters[label]}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Convert tensor to PIL image and save
            pil_image = transforms.ToPILImage()(image)
            pil_image.save(filepath)
            
    return label_counters

def create_and_sort_txt(classes, txt_dir):
    classes_txt_path = os.path.join(txt_dir, 'classes.txt')
    classes_modified_txt_path = os.path.join(txt_dir, 'class_changes.txt')
    
    for i in range(len(classes)):
        class_name = classes[str(i)].replace(' ', '')
        class_modified_name = classes[str(i)]
        with open(classes_txt_path, 'a') as file:
            file.write(class_name.lower()+'\n')
        with open(classes_modified_txt_path, 'a') as file:
            file.write(class_modified_name.lower()+'\n')

    command = ["sort", classes_modified_txt_path, "-o", classes_modified_txt_path]
    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)
    command = ["sort", classes_txt_path, "-o", classes_txt_path]
    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)
    

#%%
def process_data(dataset_name):
    datasets_dir = './datasets'
    if dataset_name == 'EuroSAT':
        load_path = "blanchon/EuroSAT_RGB"
        classes = {
            '0': 'annual crop',
            '1': 'forest',
            '2': 'herbaceous vegetation',
            '3': 'highway',
            '4': 'industrial buildings',
            '5': 'pasture',
            '6': 'permanent crop',
            '7': 'residential buildings',
            '8': 'river',
            '9': 'sealake'
            }
        output_dir_img = os.path.join(datasets_dir, dataset_name, 'images')
        if os.path.exists(output_dir_img):
            print(f"{dataset_name} dataset already downloaded")
        else:
            label_counters = {}
            split = ['train', 'validation', 'test']
            label_key = 'label'
            print(f"Downloading {dataset_name} dataset")
            for s in split:
                label_counters = save_images_from_dataloader(load_path, output_dir_img, classes, label_counters, s, label_key)
            print(f"{dataset_name} dataset downloaded")
            
            output_dir_txt = os.path.join(datasets_dir, dataset_name)
            
            create_and_sort_txt(classes, output_dir_txt)
            print("txt file created")
            
    elif dataset_name == 'OPTIMAL31':
        load_path = 'jonathan-roberts1/Optimal-31'
        classes = {'0': 'airplane',
                    '1': 'airport',
                    '2': 'baseball diamond',
                    '3': 'basketball court',
                    '4': 'beach',
                    '5': 'bridge',
                    '6': 'chaparral',
                    '7': 'church',
                    '8': 'circular farmland',
                    '9': 'commercial area',
                    '10': 'dense residential',
                    '11': 'desert',
                    '12': 'forest',
                    '13': 'freeway',
                    '14': 'golf course',
                    '15': 'ground track field',
                    '16': 'harbor',
                    '17': 'industrial area',
                    '18': 'intersection',
                    '19': 'island',
                    '20': 'lake',
                    '21': 'meadow',
                    '22': 'medium residential',
                    '23': 'mobile home park',
                    '24': 'mountain',
                    '25': 'overpass',
                    '26': 'parking lot',
                    '27': 'railway',
                    '28': 'rectangular farmland',
                    '29': 'roundabout',
                    '30': 'runway'}
        output_dir_img = os.path.join(datasets_dir, dataset_name, 'images')
        if os.path.exists(output_dir_img):
            print(f"{dataset_name} dataset already downloaded")
        else:
            label_counters = {}
            split = 'train'
            label_key = 'label'
            
            print(f"Downloading {dataset_name} dataset")
            label_counters = save_images_from_dataloader(load_path, output_dir_img, classes, label_counters, split, label_key)
            print(f"{dataset_name} dataset downloaded")
            
            output_dir_txt = os.path.join(datasets_dir, dataset_name)
            
            create_and_sort_txt(classes, output_dir_txt)
            print("txt file created")
        
    elif dataset_name == 'PatternNet':
        load_path = "blanchon/PatternNet"
        classes = { '0': 'airplane',
                    '1': 'baseball field',
                    '2': 'basketball court',
                    '3': 'beach',
                    '4': 'bridge',
                    '5': 'cemetery',
                    '6': 'chaparral',
                    '7': 'christmas tree farm',
                    '8': 'closed road',
                    '9': 'coastal mansion',
                    '10': 'crosswalk',
                    '11': 'dense residential',
                    '12': 'ferry terminal',
                    '13': 'football field',
                    '14': 'forest',
                    '15': 'freeway',
                    '16': 'golf course',
                    '17': 'harbor',
                    '18': 'intersection',
                    '19': 'mobile home park',
                    '20': 'nursing home',
                    '21': 'oil gas field',
                    '22': 'oil well',
                    '23': 'overpass',
                    '24': 'parking lot',
                    '25': 'parking space',
                    '26': 'railway',
                    '27': 'river',
                    '28': 'runway',
                    '29': 'runway marking',
                    '30': 'shipping yard',
                    '31': 'solar panel',
                    '32': 'sparse residential',
                    '33': 'storage tank',
                    '34': 'swimming pool',
                    '35': 'tennis court',
                    '36': 'transformer station',
                    '37': 'wastewater treatment plant'
                    }
        output_dir_img = os.path.join(datasets_dir, dataset_name, 'images')
        if os.path.exists(output_dir_img):
            print(f"{dataset_name} dataset already downloaded")
        else:
            label_counters = {}
            split = 'train'
            label_key = 'label'
            
            print(f"Downloading {dataset_name} dataset")
            label_counters = save_images_from_dataloader(load_path, output_dir_img, classes, label_counters, split, label_key)
            print(f"{dataset_name} dataset downloaded")
            
            output_dir_txt = os.path.join(datasets_dir, dataset_name)
            
            create_and_sort_txt(classes, output_dir_txt)
            print("txt file created")
    
    elif dataset_name == 'RESISC45':
        load_path = "timm/resisc45"
        classes = { '0': 'airplane',
                    '1': 'airport',
                    '2': 'baseball diamond',
                    '3': 'basketball court',
                    '4': 'beach',
                    '5': 'bridge',
                    '6': 'chaparral',
                    '7': 'church',
                    '8': 'circular farmland',
                    '9': 'cloud',
                    '10': 'commercial area',
                    '11': 'dense residential',
                    '12': 'desert',
                    '13': 'forest',
                    '14': 'freeway',
                    '15': 'golf course',
                    '16': 'ground track field',
                    '17': 'harbor',
                    '18': 'industrial area',
                    '19': 'intersection',
                    '20': 'island',
                    '21': 'lake',
                    '22': 'meadow',
                    '23': 'medium residential',
                    '24': 'mobile home park',
                    '25': 'mountain',
                    '26': 'overpass',
                    '27': 'palace',
                    '28': 'parking lot',
                    '29': 'railway',
                    '30': 'railway station',
                    '31': 'rectangular farmland',
                    '32': 'river',
                    '33': 'roundabout',
                    '34': 'runway',
                    '35': 'sea ice',
                    '36': 'ship',
                    '37': 'snowberg',
                    '38': 'sparse residential',
                    '39': 'stadium',
                    '40': 'storage tank',
                    '41': 'tennis court',
                    '42': 'terrace',
                    '43': 'thermal power station',
                    '44': 'wetland'
                    }
        output_dir_img = os.path.join(datasets_dir, dataset_name, 'images')
        if os.path.exists(output_dir_img):
            print(f"{dataset_name} dataset already downloaded")
        else:
            label_counters = {}
            split = ['train', 'validation', 'test']
            label_key = 'label'
            print(f"Downloading {dataset_name} dataset")
            for s in split:
                label_counters = save_images_from_dataloader(load_path, output_dir_img, classes, label_counters, s, label_key)
            print(f"{dataset_name} dataset downloaded")
            
            output_dir_txt = os.path.join(datasets_dir, dataset_name)
            
            create_and_sort_txt(classes, output_dir_txt)
            print("txt file created")
        
    elif dataset_name == 'RSC11':
        load_path = "jonathan-roberts1/RS_C11"
        classes = { '0': 'dense forest',
                    '1': 'grassland',
                    '2': 'harbor',
                    '3': 'high buildings',
                    '4': 'low buildings',
                    '5': 'overpass',
                    '6': 'railway',
                    '7': 'residential area',
                    '8': 'roads',
                    '9': 'sparse forest',
                    '10': 'storage tanks',
                    }
        output_dir_img = os.path.join(datasets_dir, dataset_name, 'images')
        if os.path.exists(output_dir_img):
            print(f"{dataset_name} dataset already downloaded")
        else:
            label_counters = {}
            split = 'train'
            label_key = 'label'
            
            print(f"Downloading {dataset_name} dataset")
            label_counters = save_images_from_dataloader(load_path, output_dir_img, classes, label_counters, split, label_key)
            print(f"{dataset_name} dataset downloaded")
            
            output_dir_txt = os.path.join(datasets_dir, dataset_name)
            
            create_and_sort_txt(classes, output_dir_txt)
            print("txt file created")
        
        
    elif dataset_name == 'RSICB256':
        load_path = "jonathan-roberts1/RSI-CB256"
        classes = { '0': 'parking lot',
                    '1': 'avenue',
                    '2': 'highway',
                    '3': 'bridge',
                    '4': 'marina',
                    '5': 'crossroads',
                    '6': 'airport runway',
                    '7': 'pipeline',
                    '8': 'town',
                    '9': 'airplane',
                    '10': 'forest',
                    '11': 'mangrove',
                    '12': 'artificial grassland',
                    '13': 'river protection forest',
                    '14': 'shrubwood',
                    '15': 'sapling',
                    '16': 'sparse forest',
                    '17': 'lakeshore',
                    '18': 'river',
                    '19': 'stream',
                    '20': 'coastline',
                    '21': 'hirst',
                    '22': 'dam',
                    '23': 'sea',
                    '24': 'snow mountain',
                    '25': 'sandbeach',
                    '26': 'mountain',
                    '27': 'desert',
                    '28': 'dry farm',
                    '29': 'green farmland',
                    '30': 'bare land',
                    '31': 'city building',
                    '32': 'residents',
                    '33': 'container',
                    '34': 'storage room'
                    }
        output_dir_img = os.path.join(datasets_dir, dataset_name, 'images')
        if os.path.exists(output_dir_img):
            print(f"{dataset_name} dataset already downloaded")
        else:
            label_counters = {}
            split = 'train'
            label_key = 'label_2'
            
            print(f"Downloading {dataset_name} dataset")
            label_counters = save_images_from_dataloader(load_path, output_dir_img, classes, label_counters, split, label_key)
            print(f"{dataset_name} dataset downloaded")
            
            output_dir_txt = os.path.join(datasets_dir, dataset_name)
            
            create_and_sort_txt(classes, output_dir_txt)
            print("txt file created")
        

        
    else: 
        print("ERROR: Wrong dataset name")
        return

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for the specified arguments')
    parser.add_argument('--dataset_name', type=str, choices=['AID', 'EuroSAT', 'OPTIMAL31', 'PatternNet', 'RESISC45', 'RSC11', 'RSICB256'], help='Name of the dataset')
    args = parser.parse_args()
    
    process_data(args.dataset_name)

