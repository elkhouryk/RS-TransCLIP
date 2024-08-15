import itertools
import torch
import argparse
from TransCLIP_solver.TransCLIP_inference import main

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run TransCLIP inference with specified parameters.")
    
    parser.add_argument('--model_names', nargs='+', default=["CLIP", "GeoRSCLIP", "RemoteCLIP", "SkyCLIP50"], help='List of model names')
    parser.add_argument('--model_architectures', nargs='+', default=["RN50", "ViT-B-32", "ViT-L-14", "ViT-H-14"], help='List of model architectures')
    parser.add_argument('--dataset_names', nargs='+', default=["AID", "EuroSAT", "MLRSNet", "OPTIMAL31", "PatternNet", "RESISC45", "RSC11", "RSICB128", "RSICB256", "WHURS19"], help='List of dataset names')
    parser.add_argument("--individualprompts", action='store_true', help="Use individual prompts instead of average")

    return parser.parse_args()

torch.cuda.set_device(0)

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.individualprompts:
        text_prompts = [
            "a satellite photo of a", "a remote sensing image of many", "a remote sensing image of a",
            "a remote sensing image of the", "a remote sensing image of the hard to see",
            "a remote sensing image of a hard to see", "a low resolution remote sensing image of the",
            "a low resolution remote sensing image of a", "a bad remote sensing image of the",
            "a bad remote sensing image of a", "a cropped remote sensing image of the",
            "a cropped remote sensing image of a", "a bright remote sensing image of the",
            "a bright remote sensing image of a", "a dark remote sensing image of the",
            "a dark remote sensing image of a", "a close-up remote sensing image of the",
            "a close-up remote sensing image of a", "a black and white remote sensing image of the",
            "a black and white remote sensing image of a", "a jpeg corrupted remote sensing image of the",
            "a jpeg corrupted remote sensing image of a", "a blurry remote sensing image of the",
            "a blurry remote sensing image of a", "a good remote sensing image of the",
            "a good remote sensing image of a", "a remote sensing image of the large",
            "a remote sensing image of a large", "a remote sensing image of the nice",
            "a remote sensing image of a nice", "a remote sensing image of the small",
            "a remote sensing image of a small", "a remote sensing image of the weird",
            "a remote sensing image of a weird", "a remote sensing image of the cool",
            "a remote sensing image of a cool", "an aerial image of many", "an aerial image of a",
            "an aerial image of the", "an aerial image of the hard to see",
            "an aerial image of a hard to see", "a low resolution aerial image of the",
            "a low resolution aerial image of a", "a bad aerial image of the",
            "a bad aerial image of a", "a cropped aerial image of the",
            "a cropped aerial image of a", "a bright aerial image of the",
            "a bright aerial image of a", "a dark aerial image of the",
            "a dark aerial image of a", "a close-up aerial image of the",
            "a close-up aerial image of a", "a black and white aerial image of the",
            "a black and white aerial image of a", "a jpeg corrupted aerial image of the",
            "a jpeg corrupted aerial image of a", "a blurry aerial image of the",
            "a blurry aerial image of a", "a good aerial image of the",
            "a good aerial image of a", "an aerial image of the large",
            "an aerial image of a large", "an aerial image of the nice",
            "an aerial image of a nice", "an aerial image of the small",
            "an aerial image of a small", "an aerial image of the weird",
            "an aerial image of a weird", "an aerial image of the cool",
            "an aerial image of a cool", "a satellite image of many",
            "a satellite image of a", "a satellite image of the",
            "a satellite image of the hard to see", "a satellite image of a hard to see",
            "a low resolution satellite image of the", "a low resolution satellite image of a",
            "a bad satellite image of the", "a bad satellite image of a",
            "a cropped satellite image of the", "a cropped satellite image of a",
            "a bright satellite image of the", "a bright satellite image of a",
            "a dark satellite image of the", "a dark satellite image of a",
            "a close-up satellite image of the", "a close-up satellite image of a",
            "a black and white satellite image of the", "a black and white satellite image of a",
            "a jpeg corrupted satellite image of the", "a jpeg corrupted satellite image of a",
            "a blurry satellite image of the", "a blurry satellite image of a",
            "a good satellite image of the", "a good satellite image of a",
            "a satellite image of the large", "a satellite image of a large",
            "a satellite image of the nice", "a satellite image of a nice",
            "a satellite image of the small", "a satellite image of a small",
            "a satellite image of the weird", "a satellite image of a weird",
            "a satellite image of the cool", "a satellite image of a cool"
        ]
        averageprompt = False
    else:
        text_prompts = ["averageprompt"]
        averageprompt = True
    
    for dataset_name in args.dataset_names:
        for model_name in args.model_names:
            for model_architecture in args.model_architectures:
                for text_prompt in text_prompts:
                    print(f"Processing: {model_name}, {model_architecture}, {dataset_name}, {text_prompt} with average prompt = {averageprompt}")
                    main(dataset_name, model_name, model_architecture, text_prompt, averageprompt)
                    torch.cuda.empty_cache()