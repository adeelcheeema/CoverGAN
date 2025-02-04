import cv2
import os

def resize_and_save_images(source_folder, target_folder, target_size):
    """
    Resizes all images in a specified folder to a target size and saves the resized images in a new folder with the same file names.

    Args:
    source_folder (str): Path to the source folder containing images.
    target_folder (str): Path to the target folder where resized images will be saved.
    target_size (tuple): A tuple of the form (width, height) specifying the new size of the images.
    """
    # Create the target folder if it does not exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all files in the source folder
    files = os.listdir(source_folder)

    # Process each file
    for file in files:
        file_path = os.path.join(source_folder, file)
        # Read the image
        image = cv2.imread(file_path)
        if image is not None:
            # Resize the image
            resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
            # Path for the resized image
            target_path = os.path.join(target_folder, file)
            # Save the resized image
            cv2.imwrite(target_path, resized_image)
        else:
            print(f"Failed to read image: {file}")

# Example usage
source_folder = 'outputs/GEN_512'
target_folder = 'outputs/GEN_256'
target_size = (256, 256)  # Target size (width, height)

resize_and_save_images(source_folder, target_folder, target_size)
