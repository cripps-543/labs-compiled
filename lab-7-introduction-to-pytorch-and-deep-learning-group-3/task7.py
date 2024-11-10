import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from utils.models import LinearNet

if __name__ == '__main__':
    # Set the batch size and number of epochs
    batch_sizes = [8192]
    num_epochs = 5  # Adjust this to your desired number of epochs

    for batch_size in batch_sizes:
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: {}".format(device))

        # Load the MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST('lab7/data', train=True, download=True, transform=transform)
        testset = datasets.MNIST('lab7/data', train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        # Create the model and optimizer
        model = LinearNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Start timer
        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Train the model
            model.train()
            train_loss = 0

            # Batch Loop
            for i, (images, labels) in enumerate(tqdm(train_loader, leave=False)):
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
                train_loss += loss.item()

            # Test the model after each epoch
            model.eval()
            test_loss = 0
            correct = 0
            total = 0

            # Batch Loop
            for images, labels in tqdm(test_loader, leave=False):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Compute the loss
                loss = F.nll_loss(outputs, labels)

                # Accumulate the loss
                test_loss += loss.item()

                # Get the predicted class
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Save the model after each epoch (optional)
            model_save_path = f"linear_net_model_epoch_{epoch + 1}_batch_{batch_size}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')

            # Print epoch statistics
            print(f"Epoch {epoch + 1} Statistics:")
            print(f'Training Loss: {train_loss / len(train_loader):.4f}, '
                  f'Test Loss: {test_loss / len(test_loader):.4f}, '
                  f'Test Accuracy: {100 * correct / total:.2f}%, '
                  f'Time Taken: {time.time() - start_time:.2f}s')

        print(f"Training complete for batch size {batch_size}.")
