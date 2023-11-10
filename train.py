import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_utils import build_model, train_model, save_checkpoint

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint.')
    parser.add_argument('data_dir', type=str, help='Path to the dataset directory')
    parser.add_argument('--save_dir', type=str, default='save_directory', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13', help='Architecture (e.g., "vgg13")')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()

    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load the datasets
    image_datasets = {x: datasets.ImageFolder(args.data_dir, transform=data_transforms[x]) for x in ['train', 'valid', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}

    # Build the model
    model = build_model(args.arch, args.hidden_units)

    # Train the model
    train_model(model, dataloaders['train'], dataloaders['valid'], args.learning_rate, args.epochs, args.gpu)

    # Save the checkpoint
    save_checkpoint(model, image_datasets['train'].class_to_idx, args.arch, args.hidden_units, args.save_dir)

if __name__ == "__main__":
    main()
