#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import argparse
import libarchive
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
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing images")):
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
    datasets_dir = '/Users/tgodelaine/Desktop/ICASSP/data' #'./datasets'
    if dataset_name == 'AID':
        load_path = "jonathan-roberts1/Million-AID"
        classes = {
            '0': 'dam',
            '1': 'religious land',
            '2': 'rock land',
            '3': 'sparse shrub land',
            '4': 'arable land',
            '5': 'factory area',
            '6': 'detached house',
            '7': 'desert',
            '8': 'lake',
            '9': 'power station',
            '10': 'beach',
            '11': 'ice land',
            '12': 'bare land',
            '13': 'island',
            '14': 'woodland',
            '15': 'mobile home park',
            '16': 'railway area',
            '17': 'river',
            '18': 'grassland',
            '19': 'apartment',
            '20': 'special land',
            '21': 'port area',
            '22': 'commercial area',
            '23': 'highway area',
            '24': 'mining area',
            '25': 'sports land',
            '26': 'airport area',
            '27': 'leisure land'
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
        
    elif dataset_name == 'EuroSAT':
        load_path = "blanchon/EuroSAT_RGB"
        classes = {
            '0': 'annual crop',
            '1': 'forest',
            '2': 'herbaceous vegetation',
            '3': 'highway',
            '4': 'industrial buildings',
            '5': 'pasture',
            '6': 'permanent crop',
            '7': 'Rrsidential buildings',
            '8': 'river',
            '9': 'seaLake'
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
            
    elif dataset_name == 'MLRSNet':
        dataset_dir_init = os.path.join(datasets_dir, 'MLRSNet_Dataset/images')
        dataset_dir = os.path.join(datasets_dir, dataset_name, 'images')
        
        if os.path.exists(dataset_dir):
            print(f"{dataset_name} dataset already downloaded")
        else:
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            
            classes_folder = os.listdir(dataset_dir_init)
            classes_folder = [c for c in classes_folder if c != '.DS_Store']
            classes_folder = [c for c in classes_folder if 'rar' not in c]

            print(f"Downloading {dataset_name} dataset")
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
        
            shutil.rmtree(dataset_dir_init)
            print(f"Directory {dataset_dir_init} has been removed")
            
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
        
    elif dataset_name == 'RSICB128':
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
        
    elif dataset_name == 'WHURS19':
        dataset_dir_init = os.path.join(datasets_dir, 'WHURS19_Dataset') 
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
                
                for i, img in enumerate(tqdm(images,  desc="Processing images", leave=False)): 
                    image_path = os.path.join(dataset_dir_init, c, img)
                    
                    new_image_name = img.lower() 
                    if '-' in new_image_name:
                        new_image_name = new_image_name.replace('-', '_')
                    image_path_modified = os.path.join(dataset_dir, new_image_name)
    
                    command = ["mv", image_path, image_path_modified]
                    # Execute the command
                    result = subprocess.run(command, capture_output=True, text=True)
            print(f"{dataset_name} dataset downloaded")
            
            classes = {'0': 'airport',
                       '1': 'beach',
                       '2': 'bridge',
                       '3': 'commercial',
                       '4': 'desert',
                       '5': 'farmland',
                       '6': 'football field',
                       '7': 'forest',
                       '8': 'industrial',
                       '9': 'meadow',
                       '10': 'mountain',
                       '11': 'park',
                       '12': 'parking',
                       '13': 'pond',
                       '14': 'port',
                       '15': 'railway station',
                       '16': 'residential',
                       '17': 'river',
                       '18': 'viaduct'
                       }
            output_dir_txt = os.path.join(datasets_dir, dataset_name)
            create_and_sort_txt(classes, output_dir_txt)
            print("txt file created")
            
            shutil.rmtree(dataset_dir_init)
            print(f"Directory {dataset_dir_init} has been removed")
        
    else: 
        print("ERROR: Wrong dataset name")
        return

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for the specified arguments')
    parser.add_argument('--dataset_name', type=str, choices=['AID', 'EuroSat', 'MLRSNet', 'OPTIMAL31', 'PatternNet', 'RESISC45', 'RSC11', 'RSICB128', 'RSICB256', 'WHURS19'], help='Name of the dataset')
    args = parser.parse_args()
    
    process_data(args.dataset_name)

