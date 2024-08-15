import argparse
from feature_generator.featuregeneration import main

def run_featuregeneration(args):
    # Set text_prompts to a single prompt if image_fg is true
    if args.image_fg:
        args.text_prompts = ["a satellite photo of a"]

    # Iterate over each combination of model name and architecture
    for model_name in args.model_names:
        for model_architecture in args.model_architectures:
            for dataset_name in args.dataset_names:
                for text_prompt in args.text_prompts:
                    if args.image_fg:
                        print(f"Running feature generation for model: {model_name}, architecture: {model_architecture} on dataset {dataset_name}")
                    else:
                        print(f"Running feature generation for model: {model_name}, architecture: {model_architecture} on dataset {dataset_name} for text prompt {text_prompt}")
                    try:
                        main(args.results_dir, args.datasets_dir, dataset_name, model_name, model_architecture, args.image_batch_size, args.image_fg, args.text_fg, args.gpu_id, args.num_workers, text_prompt)
                    except Exception as e:
                        print(f"An error occurred for model: {model_name}, architecture: {model_architecture} on dataset {dataset_name}")
                        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inferences with specified parameters.')
    
    # Define mutually exclusive group for inference type
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_fg', action='store_true', help='Set to run image feature generation')
    group.add_argument('--text_fg', action='store_true', help='Set to run text feature generation')
    
    # Define other arguments
    parser.add_argument('--results_dir', type=str, default='results', help='Directory for results')
    parser.add_argument('--datasets_dir', type=str, default='datasets', help='Directory for datasets')
    parser.add_argument('--image_batch_size', type=int, default=64, help='Batch size for images')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers')
    parser.add_argument('--model_names', nargs='+', default=["CLIP", "GeoRSCLIP", "RemoteCLIP", "SkyCLIP50"], help='List of model names')
    parser.add_argument('--model_architectures', nargs='+', default=["RN50", "ViT-B-32", "ViT-L-14", "ViT-H-14"], help='List of model architectures')
    parser.add_argument('--dataset_names', nargs='+', default=["AID", "EuroSAT", "MLRSNet", "OPTIMAL31", "PatternNet", "RESISC45", "RSC11", "RSICB128", "RSICB256", "WHURS19"], help='List of dataset names')
    parser.add_argument('--text_prompts', nargs='+', default=[
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
        "a low resolution aerial image of a", "a bad aerial image of the", "a bad aerial image of a", 
        "a cropped aerial image of the", "a cropped aerial image of a", "a bright aerial image of the", 
        "a bright aerial image of a", "a dark aerial image of the", "a dark aerial image of a", 
        "a close-up aerial image of the", "a close-up aerial image of a", 
        "a black and white aerial image of the", "a black and white aerial image of a", 
        "a jpeg corrupted aerial image of the", "a jpeg corrupted aerial image of a", 
        "a blurry aerial image of the", "a blurry aerial image of a", "a good aerial image of the", 
        "a good aerial image of a", "an aerial image of the large", "an aerial image of a large", 
        "an aerial image of the nice", "an aerial image of a nice", "an aerial image of the small", 
        "an aerial image of a small", "an aerial image of the weird", "an aerial image of a weird", 
        "an aerial image of the cool", "an aerial image of a cool", "a satellite image of many", 
        "a satellite image of a", "a satellite image of the", "a satellite image of the hard to see", 
        "a satellite image of a hard to see", "a low resolution satellite image of the", 
        "a low resolution satellite image of a", "a bad satellite image of the", 
        "a bad satellite image of a", "a cropped satellite image of the", 
        "a cropped satellite image of a", "a bright satellite image of the", 
        "a bright satellite image of a", "a dark satellite image of the", 
        "a dark satellite image of a", "a close-up satellite image of the", 
        "a close-up satellite image of a", "a black and white satellite image of the", 
        "a black and white satellite image of a", "a jpeg corrupted satellite image of the", 
        "a jpeg corrupted satellite image of a", "a blurry satellite image of the", 
        "a blurry satellite image of a", "a good satellite image of the", 
        "a good satellite image of a", "a satellite image of the large", 
        "a satellite image of a large", "a satellite image of the nice", 
        "a satellite image of a nice", "a satellite image of the small", 
        "a satellite image of a small", "a satellite image of the weird", 
        "a satellite image of a weird", "a satellite image of the cool", 
        "a satellite image of a cool"
    ], help='List of text prompts')

    args = parser.parse_args()
    run_featuregeneration(args)