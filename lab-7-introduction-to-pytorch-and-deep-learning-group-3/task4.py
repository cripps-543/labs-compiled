import torch
from torchvision import datasets, transforms
import cv2
import numpy as np
import random

if __name__ == '__main__':

    # Load the MNIST dataset
    transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    testset = datasets.MNIST('lab7/data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    # Number of random images to display
    num_images = 5

    # Randomly select indices
    indices = random.sample(range(len(testset)), num_images)

    for index in indices:
        # Get the image and label
        image, label = testset[index]

        # Convert the image tensor to a NumPy array
        image = image.numpy().squeeze()  # Remove channel dimension

        # Rescale the image to 0-255 for OpenCV
        image = (image * 255).astype(np.uint8)

        # Display the image with OpenCV
        cv2.imshow(f'Random image', image)

        # Wait for a key press to close the image window
        cv2.waitKey(0)

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()