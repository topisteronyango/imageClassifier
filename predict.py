import argparse
import json
from PIL import Image
import torch
from model_utils import load_checkpoint, predict

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict flower name from an image with the probability of that name.')
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=3, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    # Load the category names mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load the checkpoint
    model, class_to_idx = load_checkpoint(args.checkpoint)

    # Process the image
    image = Image.open(args.input)
    processed_image = predict.process_image(image)

    # Make predictions
    probs, classes = predict.predict(processed_image, model, class_to_idx, args.top_k, args.gpu)

    # Convert class indices to class names
    class_names = [cat_to_name[class_label] for class_label in classes]

    # Print the results
    for prob, class_name in zip(probs, class_names):
        print(f'Class: {class_name}, Probability: {prob:.4f}')

if __name__ == "__main__":
    main()
