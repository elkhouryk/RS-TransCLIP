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
    if dataset_name == 'RSICB128':
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
                '36':'sea',
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
            
            classes = {'0': 'airplane',
                       '1': 'airport',
                       '2': 'bare soil',
                       '3': 'baseball diamond',
                       '4': 'basketball court',
                       '5': 'beach',
                       '6': 'bridge',
                       '7': 'buildings',
                       '8': 'cars',
                       '9': 'chaparral',
                       '10': 'cloud',
                       '11': 'containers',
                       '12': 'crosswalk',
                       '13': 'dense residential area',
                       '14': 'desert',
                       '15': 'dock',
                       '16': 'factory',
                       '17': 'field',
                       '18': 'football field',
                       '19': 'forest',
                       '20': 'freeway',
                       '21': 'golf course',
                       '22': 'grass',
                       '23': 'greenhouse',
                       '24': 'gully',
                       '25': 'habor',
                       '26': 'intersection',
                       '27': 'island',
                       '28': 'lake',
                       '29': 'mobile home',
                       '30': 'mountain',
                       '31': 'overpass',
                       '32': 'park',
                       '33': 'parking lot',
                       '34': 'parkway',
                       '35': 'pavement',
                       '36': 'railway',
                       '37': 'railway station',
                       '38': 'river',
                       '39': 'road',
                       '40': 'roundabout',
                       '41': 'runway',
                       '42': 'sand',
                       '43': 'sea',
                       '44': 'ships',
                       '45': 'snow',
                       '46': 'snowberg',
                       '47': 'sparse residential area',
                       '48': 'stadium',
                       '49': 'swimming pool',
                       '50': 'tanks',
                       '51': 'tennis court',
                       '52': 'terrace',
                       '53': 'track',
                       '54': 'trail',
                       '55': 'transmission tower',
                       '56': 'trees',
                       '57': 'water',
                       '58': 'wetland',
                       '59': 'wind turbine'}
            
            output_dir_txt = os.path.join(datasets_dir, dataset_name)
            
            create_and_sort_txt(classes, output_dir_txt)
            print("txt file created")  
        
            #shutil.rmtree(os.path.join(datasets_dir, 'MLRSNet_Dataset/'))
            #print(f"Directory {dataset_dir_init} has been removed")
    else: 
        print("ERROR: Wrong dataset name")
        return


#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for the specified arguments')
    parser.add_argument('--dataset_name', type=str, choices=['MLRSNet', 'RSICB128'], help='Name of the dataset')
    args = parser.parse_args()
    
    process_data(args.dataset_name)

    
