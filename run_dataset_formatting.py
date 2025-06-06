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
import zipfile

#%%
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

def process_data(dataset_name):
    datasets_dir = './datasets'
    if dataset_name == 'AID':
        # Extract the zip file
        zip_file_path = os.path.join(datasets_dir, 'AID.zip')
        extract_to_path = os.path.join(datasets_dir, 'AID_Dataset')
        os.makedirs(extract_to_path, exist_ok=True)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        os.remove(zip_file_path)
        print(f'Extracted to {extract_to_path}')
        
        dataset_dir_init = os.path.join(datasets_dir, 'AID_Dataset')
        dataset_dir = os.path.join(datasets_dir, dataset_name, 'images')
        if os.path.exists(dataset_dir):
            print(f"{dataset_name} dataset already downloaded")
        else:
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
    
            classes_folder = os.listdir(dataset_dir_init)
            classes_folder = [c for c in classes_folder if c != '.DS_Store']
            print(f"Downloading {dataset_name} dataset")
            for j, c in enumerate(classes_folder):
                images = os.listdir(os.path.join(dataset_dir_init, c))
                images = [i for i in images if i != '.DS_Store']
                
                #for i, img in enumerate(images):
                for i, img in enumerate(tqdm(images, desc=f"Processing images in {c}", leave=False)): 
                    image_path = os.path.join(dataset_dir_init, c, img)
                    
                    c_split = c.split("_")  # Class name split by underscores
                    class_name = ''.join(c_split)  # Combine class name parts
                    img_split = img.split("(")[-1].split(")")[0]  # Extract numeric part from image name
                    
                    # Check if the class name is already in the image name
                    if class_name.lower() in img.lower():
                        new_image_name = img  # Keep the original image name
                    else:
                        new_image_name = f"{class_name}_{img_split}.tif"  # Add class name and numeric part
                    
                    image_path_modified = os.path.join(dataset_dir, new_image_name)

                    # Move the image to the new location with the renamed file
                    command = ["mv", image_path, image_path_modified]
                    result = subprocess.run(command, capture_output=True, text=True)
            print(f"{dataset_name} dataset downloaded")
        
            classes = {
                '0':'Airport',
                '1':'Bare Land',
                '2':'Baseball Field',
                '3':'Beach',
                '4':'Bridge',
                '5':'Center',
                '6':'Church',
                '7':'Commercial',
                '8':'Dense Residential',
                '9':'Desert',
                '10':'Farmland',
                '11':'Forest',
                '12':"Industrial",
                '13':"Meadow",
                '14':'Medium Residential',
                '15':'Mountain',
                '16':'Park',
                '17':'Parking',
                '18':'Playground',
                '19':'Pond',
                '20':'Port',
                '21':'Railway Station',
                '22':'Resort',
                '23':'River',
                '24':'School',
                '25':'Sparse Residential',
                '26':'Square',
                '27':'Stadium',
                '28':'Storage Tanks',
                '29':'Viaduct',
                }
            output_dir_txt = os.path.join(datasets_dir, dataset_name) 
            create_and_sort_txt(classes, output_dir_txt)
            print("txt file created")
            
            shutil.rmtree(dataset_dir_init)
            print(f"Directory {dataset_dir_init} has been removed")
    elif dataset_name == 'MLRSNet':
        # Extract the zip file
        zip_file_path = os.path.join(datasets_dir, 'MLRSNet.zip')
        extract_to_path = os.path.join(datasets_dir, 'MLRSNet_Dataset')
        os.makedirs(extract_to_path, exist_ok=True)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        os.remove(zip_file_path)
        print(f'Extracted to {extract_to_path}')
        
        dataset_dir_init = os.path.join(datasets_dir, 'MLRSNet_Dataset/Images')
        dataset_dir = os.path.join(datasets_dir, dataset_name, 'images')
        
        if os.path.exists(dataset_dir):
            print(f"{dataset_name} dataset already downloaded")
        else:
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            
            classes_folder = os.listdir(dataset_dir_init)
            classes_folder = [c for c in classes_folder if c != '.DS_Store']
            for j, c in enumerate(classes_folder):
                images = os.listdir(os.path.join(dataset_dir_init, c))
                images = [i for i in images if i != '.DS_Store']
                
                for i, img in enumerate(tqdm(images, desc=f"Processing images in {c}", leave=False)): 
                    image_path = os.path.join(dataset_dir_init, c, img)
                    
                    new_image_name = (''.join(img.split("_")[:-1])).lower() + "_" + str(i) + '.jpg'
                    
                    image_path_modified = os.path.join(dataset_dir, new_image_name)

                    command = ["mv", image_path, image_path_modified]
                    # Execute the command
                    result = subprocess.run(command, capture_output=True, text=True)
            print(f"{dataset_name} dataset downloaded")
            

            classes = {
                    '0': 'airplane',
                    '1': 'airport',
                    '2': 'bare land',
                    '3': 'baseball diamond',
                    '4': 'basketball court',
                    '5': 'beach',
                    '6': 'bridge',
                    '7': 'chaparral',
                    '8': 'cloud',
                    '9': 'commercial area',
                    '10': 'dense residential area',
                    '11': 'desert',
                    '12': 'eroded farmland',
                    '13': 'farmland',
                    '14': 'forest',
                    '15': 'freeway',
                    '16': 'golf course',
                    '17': 'ground track field',
                    '18': 'harbor and port',
                    '19': 'industrial area',
                    '20': 'intersection',
                    '21': 'island',
                    '22': 'lake',
                    '23': 'mobile home park',
                    '24': 'meadow',
                    '25': 'mountain',
                    '26': 'overpass',
                    '27': 'park',
                    '28': 'parking lot',
                    '29': 'parkway',
                    '30': 'railway',
                    '31': 'railway station',
                    '32': 'river',
                    '33': 'roundabout',
                    '34': 'shipping yard',
                    '35': 'snowberg',
                    '36': 'sparse residential area',
                    '37': 'stadium',
                    '38': 'storage tank',
                    '39': 'swimmimg pool',
                    '40': 'tennis court',
                    '41': 'terrace',
                    '42': 'transmission tower',
                    '43': 'vegetable greenhouse',
                    '44': 'wetland',
                    '45': 'wind turbine'}
            
            
            output_dir_txt = os.path.join(datasets_dir, dataset_name)
            
            create_and_sort_txt(classes, output_dir_txt)
            print("txt file created")  
        
            shutil.rmtree(os.path.join(datasets_dir, 'MLRSNet_Dataset/'))
            print(f"Directory {dataset_dir_init} has been removed")
    elif dataset_name == 'RSICB128':
        # Extract the zip file
        zip_file_path = os.path.join(datasets_dir, 'RSI-CB128 Dataset.zip')
        extract_to_path = os.path.join(datasets_dir, 'RSICB128_Dataset')
        os.makedirs(extract_to_path, exist_ok=True)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        os.remove(zip_file_path)
        print(f'Extracted to {extract_to_path}')
        
        dataset_dir_init = os.path.join(datasets_dir, 'RSICB128_Dataset')
        dataset_dir = os.path.join(datasets_dir, dataset_name, 'images')
        if os.path.exists(dataset_dir):
            print(f"{dataset_name} dataset already downloaded")
        else:
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
    
            classes_folder = os.listdir(dataset_dir_init)
            classes_folder = [c for c in classes_folder if c != '.DS_Store']
            print(f"Downloading {dataset_name} dataset")
            for j, c in enumerate(classes_folder):
                images = os.listdir(os.path.join(dataset_dir_init, c))
                images = [i for i in images if i != '.DS_Store']
                
                #for i, img in enumerate(images):
                for i, img in enumerate(tqdm(images, desc=f"Processing images in {c}", leave=False)): 
                    image_path = os.path.join(dataset_dir_init, c, img)
                    
                    c_split = c.split("_")
                    img_split = img.split("(")[-1].split(")")[0]
                    
                    new_image_name = str(''.join(c_split)) + "_" + str(img_split) + '.tif'
                    image_path_modified = os.path.join(dataset_dir, new_image_name)
    
                    command = ["mv", image_path, image_path_modified]
                    # Execute the command
                    result = subprocess.run(command, capture_output=True, text=True)
            print(f"{dataset_name} dataset downloaded")
        
            classes = {
                '0':'airport runway',
                '1':'artificial grassland',
                '2':'avenue',
                '3':'bare land',
                '4':'bridge',
                '5':'city avenue',
                '6':'city building',
                '7':'city green tree',
                '8':'city road',
                '9':'coastline',
                '10':'container',
                '11':'crossroads',
                '12':"dam",
                '13':"desert",
                '14':'dry farm',
                '15':'forest',
                '16':'fork road',
                '17':'grave',
                '18':'green farmland',
                '19':'highway',
                '20':'hirst',
                '21':'lakeshore',
                '22':'mangrove',
                '23':'marina',
                '24':'mountain',
                '25':'mountain road',
                '26':'natural grassland',
                '27':'overpass',
                '28':'parkinglot',
                '29':'pipeline',
                '30':'rail',
                '31':'residents',
                '32':'river',
                '33':'river protection forest',
                '34':'sandbeach',
                '35':'sapling',
                '36':'sea or lake',
                '37':'shrubwood',
                '38':'snow mountain',
                '39':'sparse forest',
                '40':'storage room',
                '41':'stream',
                '42':'tower',
                '43':'town',
                '44':'turning circle'
                }
            output_dir_txt = os.path.join(datasets_dir, dataset_name) 
            create_and_sort_txt(classes, output_dir_txt)
            print("txt file created")
            
            shutil.rmtree(dataset_dir_init)
            print(f"Directory {dataset_dir_init} has been removed")
    else: 
        print("ERROR: Wrong dataset name")
        return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for the specified arguments')
    parser.add_argument('--dataset_name', type=str, choices=['AID','MLRSNet','RSICB128'], help='Name of the dataset')
    args = parser.parse_args()
    
    process_data(args.dataset_name)

    
