import argparse
from averageprompt_generator.averageprompt import main

def run_averageprompt(args):
    # Iterate over each combination and call main
    for dataset_name in args.dataset_names:
        for model_name in args.model_names:
            for model_architecture in args.model_architectures:
                main(dataset_name, model_name, model_architecture)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run average prompt generation.')
    
    # Define arguments
    parser.add_argument('--model_names', nargs='+', default=["CLIP", "GeoRSCLIP", "RemoteCLIP", "SkyCLIP50"], help='List of model names')
    parser.add_argument('--model_architectures', nargs='+', default=["RN50", "ViT-B-32", "ViT-L-14", "ViT-H-14"], help='List of model architectures')
    parser.add_argument('--dataset_names', nargs='+', default=["AID", "EuroSAT", "MLRSNet", "OPTIMAL31", "PatternNet", "RESISC45", "RSC11", "RSICB128", "RSICB256", "WHURS19"], help='List of dataset names')

    args = parser.parse_args()
    run_averageprompt(args)