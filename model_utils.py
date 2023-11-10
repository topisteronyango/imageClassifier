import torch
from torchvision import models
from torch import nn, optim

def build_model(arch='vgg13', hidden_units=512):
    # Load a pre-trained model
    model = getattr(models, arch)(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
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

# def save_checkpoint(model, class_to_idx, arch, hidden_units, save_dir='checkpoint'):
#     checkpoint = {
#         'arch': arch,
#         'hidden_units': hidden_units,
#         'state_dict': model.state_dict(),
#         'class_to_idx': class_to_idx
#     }
    
#     os.makedirs(save_dir, exist_ok=True)


# #     torch.save(checkpoint, f'{save_dir}/checkpoint.pth')
#     torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))

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
    # Load the checkpoint from the specified file path
    checkpoint = torch.load("checkpoint.pth")

    # Example: Load model architecture, state_dict, and other information
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def predict(image_path, model, topk=5):
    model.eval()
    
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        if gpu:
            model.cuda()
            image = image.cuda()
        
        output = model(image)
        
        probabilities, classes = torch.topk(output, topk)
        probabilities = torch.nn.functional.softmax(probabilities[0], dim=0).numpy()
        classes = [model.class_to_idx[idx] for idx in classes[0].cpu().numpy()]
        
    return probabilities, classes

