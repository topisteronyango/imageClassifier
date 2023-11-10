import torch
from torchvision import models
from torch import nn, optim

from collections import OrderedDict

def build_model(arch='vgg16', hidden_units=512):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        # Modify the classifier
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        # Modify the classifier
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    
    else:
        raise ValueError("Architecture not supported. Please choose 'vgg16' or 'densenet121'")
    
    # Update the model's classifier
    model.classifier = classifier
    
    return model

def train_model(model, train_loader, valid_loader, learning_rate=0.01, epochs=5, gpu=False, batch_size=32):

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    device = torch.device('cuda:0' if gpu and torch.cuda.is_available() else 'cpu')

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

        model.eval()
        validation_loss = 0.0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                validation_loss += loss.item()

                # Calculate accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f'Validation Loss: {validation_loss/len(valid_loader)}, Accuracy: {accuracy/len(valid_loader)}')



def save_checkpoint(model, class_to_idx, arch, hidden_units):
    os.makedirs(save_dir, exist_ok=True)
    
#     checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')

    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx
    }
    
    torch.save(checkpoint, 'checkpoint.pth')


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError("Architecture not supported. Please choose 'vgg16' or 'densenet121'")
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Load the model state dict
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model, model.class_to_idx

def predict(image_path, model, topk=5):
    # Process the image
    processed_image = process_image(image_path)
    processed_image = torch.from_numpy(processed_image).float().unsqueeze(0)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Make the prediction
    with torch.no_grad():
        output = model(processed_image)
    
    # Get the top probabilities and classes
    probabilities, classes = torch.topk(output, topk)
    
    # Convert indices to class labels
    idx_to_class = {idx: class_label for class_label, idx in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in classes[0]]
    
    # Convert probabilities to numpy array
    probabilities = probabilities.exp().numpy()
    
    return probabilities[0], top_classes


