import torch
from torchvision import datasets, transforms
import cv2
import numpy as np
import random

from utils.models import LinearNet  # Ensure you have the correct import for your model class

if __name__ == '__main__':
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    testset = datasets.MNIST('lab7/data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    # Number of random images to display
    num_images = 5

    model = LinearNet()
    model_save_path = "t9_linear_net_lowest_loss_model_batch_8192.pth"
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # Randomly select indices
    indices = random.sample(range(len(testset)), num_images)

    for index in indices:
        # Get the image and label
        image, label = testset[index]


        image = image.unsqueeze(0).to(torch.device("cpu"))

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)

        # Convert the image tensor to a NumPy array
        image = image.squeeze().cpu().numpy()  # Remove channel dimension
        # Rescale the image to 0-255 for OpenCV
        image = (image * 255).astype(np.uint8)

        # Print predicted and actual values
        predicted_value = predicted.item()
        actual_value = label
        is_correct = predicted_value == actual_value
        
        # Display the image with OpenCV
        cv2.imshow(f'Predicted: {predicted_value}, Actual: {actual_value}, Correct: {is_correct}', image)

        
        # Display result in console
        print(f'Predicted: {predicted_value}, Actual: {actual_value}, Correct: {is_correct}')

        # Wait for a key press to close the image window
        cv2.waitKey(0)

        # Destroy all OpenCV windows
        cv2.destroyAllWindows()
