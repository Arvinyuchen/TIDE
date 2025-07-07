import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm

import tide.data.datasets as td

# One iteration over the entire training dataset
# Runs the model on the training data
# Computes the loss
# Updates the model weights
# Tracks the training accuracy and average loss
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0 
    total = 0

    # Loops through all batches (hardcoded batch size) in the dataset
    # tqdm shows progress bar during training
    # images [batch_size, C, H, W]
    for images, labels, domain_ids in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        # Clears the old gradients from the previous batch
        optimizer.zero_grad()
        # outputs [batch_size, num_classes]
        outputs = model(images)
        loss = criterion(outputs, labels)
        # backpropagation to compute the gradients of the loss
        # with respect to the model's parameters
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct /total 
    return avg_loss, accuracy


# Evaluating the model on dataloaders 
@torch.no_grad()
def evaluate(model, dataloaders, device):
    model.eval()
    results = {}

    for domain, loader in dataloaders.items():
        correct = 0
        total = 0

        for images, labels, domain_ids in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += images.size(0)

        accuracy = correct / total
        results[domain] = accuracy

    return results



def train_single_source_erm(dataset_name, source_domain, num_classes=7, num_epochs=10, batch_size=64, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare Dataset and Dataloader
    # Image transformation
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    # Source domain dataset and dataloader
    train_ds = td.get_dataset(dataset_name, 
                              root= f"./data/{dataset_name}", 
                              split="train", 
                              domain=source_domain, 
                              transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)

    all_domains = td.get_dataset_domains(dataset_name)
    target_domains = [d for d in all_domains if d != source_domain]

    # Build the source test loader
    source_test_ds = td.get_dataset(dataset_name, 
                              root= f"./data/{dataset_name}", 
                              split="test", 
                              domain=source_domain, 
                              transform=transform)
    source_test_loader = DataLoader(source_test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Build the target domain test data loaders
    target_loaders = {}
    for domain in target_domains:
        ds = td.get_dataset(dataset_name, root=f"./data/{dataset_name}", split="test" ,domain=domain, transform=transform)
        target_loaders[domain] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    

    # Prepare Model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes) # PACS has 7 classes
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()


    # Training Loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Accuracy={train_acc:.4f}")

        # Evaluate on source domains test set
        print("\nEvaluating on Source Domain Test Set:")
        source_results = evaluate(model, {source_domain: source_test_loader}, device)
        print(f"Source Domain ({source_domain}) Test Accuracy: {source_results[source_domain] * 100:.2f}%")
        
        # Evaluate on target domains
        print("\nEvaluating on Unseen Target Domains:")
        results = evaluate(model, target_loaders, device)
        for domain, acc in results.items():
            print(f"{domain}: {acc * 100:.2f}%")
    
    return model

