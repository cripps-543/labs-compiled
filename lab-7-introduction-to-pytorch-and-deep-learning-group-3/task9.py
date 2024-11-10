import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from utils.models import LinearNet
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Set the batch size and number of epochs
    batch_sizes = [8192]
    num_epochs = 3  # Adjust this to your desired number of epochs

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

        # Variables to track the lowest training loss and losses over epochs
        lowest_train_loss = float('inf')
        lowest_loss_model_path = ""
        train_losses = []
        test_losses = []

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

            # Average training loss for the epoch
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Check if the current training loss is lower than the lowest recorded loss
            if avg_train_loss < lowest_train_loss:
                lowest_train_loss = avg_train_loss
                lowest_loss_model_path = f"t9_linear_net_lowest_loss_model_batch_{batch_size}.pth"
                torch.save(model.state_dict(), lowest_loss_model_path)
                print(f'Model with lowest training loss saved to {lowest_loss_model_path}')

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

            # Average test loss for the epoch
            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)

            # Print epoch statistics
            print(f"Epoch {epoch + 1} Statistics:")
            print(f'Training Loss: {avg_train_loss:.4f}, '
                  f'Test Loss: {avg_test_loss:.4f}, '
                  f'Test Accuracy: {100 * correct / total:.2f}%, '
                  f'Time Taken: {time.time() - start_time:.2f}s')

            # Save the training and test losses graph
            plt.figure()
            plt.plot(range(1, epoch + 2), train_losses, label='Training Loss', color='blue', marker='.')
            plt.plot(range(1, epoch + 2), test_losses, label='Test Loss', color='orange', marker='*')
            plt.title('Training and Test Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid()
            plt.savefig(f'training_test_loss_epoch_batch_{batch_size}.png')
            plt.close()


        # Save the model from the most recent epoch
        model_save_path = f"T9_linear_net_model_epoch_{num_epochs}_batch_{batch_size}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')
        print(f"Training complete for batch size {batch_size}.")
