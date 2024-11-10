import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from pyexpat import model
from torchvision import datasets, transforms
from tqdm import tqdm
from utils.models import LinearNet

if __name__ == '__main__':

    # Set the batch size
    batch_sizes = [8192]
    for batch_size in batch_sizes:
        # Check if CUDA is available (Note: It is not currently benificial to use GPU acceleration of the Raspberry Pi)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print("Device: {}".format(device))

        # Load the MNIST dataset
        transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = datasets.MNIST('lab7/data', train=True, download=True, transform=transform)
        testset = datasets.MNIST('lab7/data', train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        # Start timer
        t = time.time_ns()

        # Create the model and optimizer
        model = LinearNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train the model
        model.train()
        train_loss = 0

        # Batch Loop
        for i, (images, labels) in enumerate(tqdm(train_loader, leave=False)):

            # Move the data to the device (CPU or GPU)
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            if i % 500 == 0:
                print(outputs.shape, labels.shape)

            # Compute the loss
            loss = F.nll_loss(outputs, labels)

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Accumulate the loss
            train_loss = train_loss + loss.item()

        # Test the model
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        # Batch Loop
        for images, labels in tqdm(test_loader, leave=False):

            # Move the data to the device (CPU or GPU)
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = F.nll_loss(outputs, labels)

            # Accumulate the loss
            test_loss = test_loss + loss.item()

            # Get the predicted class from the maximum value in the output-list of class scores
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            # Accumulate the number of correct classifications
            correct += (predicted == labels).sum().item()

        # Print the epoch statistics
        print(f"BATCH SIZE: {batch_size} \n")
        print(f'Training Loss: {train_loss / len(train_loader):.4f}, Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%, Time Taken: {(time.time_ns() - t) / 1e9:.2f}s')