import cv2
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.ndimage import uniform_filter


def load_block_data(csv_path):
    """
    Loads the Minecraft block data from a CSV and prepares for color matching.
    :param csv_path: Path to the CSV file containing block data.
    :return: List of block paths, KDTree for color matching, and colors in RGB format.
    """
    block_data = pd.read_csv(csv_path)
    colors = block_data[["red", "green", "blue"]].values  # RGB values
    block_paths = block_data["image_path"].tolist()  # Image file paths
    color_tree = KDTree(colors)  # KDTree for fast nearest neighbor search
    return block_paths, color_tree, colors, block_data


def find_closest_texture(avg_color, block_paths, color_tree, block_data):
    """
    Finds the closest Minecraft block texture based on average color.
    :param avg_color: (R, G, B) tuple representing the average color.
    :param block_paths: List of block image paths.
    :param color_tree: KDTree for nearest neighbor search.
    :param block_data: DataFrame containing block metadata.
    :return: Path to the closest block texture.
    """
    # Convert avg_color to a NumPy array for comparison
    avg_color = np.array(avg_color)

    # Find the nearest neighbor using KDTree
    _, idx = color_tree.query(avg_color)

    return block_paths[idx]


def pixelated_image(image, block_size):
    """
    Creates a pixelated version of the input image.
    :param image: Input image (BGR format).
    :param block_size: Size of each pixel block.
    :return: Pixelated image.
    """
    height, width, _ = image.shape
    pixelated = np.zeros_like(image)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Determine the dimensions of the current block
            block_height = min(block_size, height - y)
            block_width = min(block_size, width - x)

            # Extract the block and compute its average color
            block = image[y : y + block_height, x : x + block_width]
            avg_color_bgr = block.mean(axis=(0, 1)).astype(int)
            avg_color_rgb = avg_color_bgr[::-1]  # Convert BGR to RGB

            # Fill the block with its average color
            pixelated[y : y + block_height, x : x + block_width] = avg_color_bgr

    return pixelated


def pixelated_minecraft_mosaic(image, block_size, csv_path):
    """
    Creates a pixelated Minecraft mosaic by replacing blocks with textures.
    :param image: Input image (BGR format).
    :param block_size: Size of each pixel block.
    :param csv_path: Path to the CSV file containing block data.
    :return: Mosaic image.
    """
    height, width, _ = image.shape
    output = np.zeros((height, width, 3), dtype=np.uint8)  # Output image

    # Load block data and prepare KDTree
    block_paths, color_tree, colors, block_data = load_block_data(csv_path)

    # Loop over blocks in the grid
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Determine the dimensions of the current block
            block_height = min(block_size, height - y)
            block_width = min(block_size, width - x)

            # Extract the block and compute its average color
            block = image[y : y + block_height, x : x + block_width]
            avg_color_bgr = block.mean(axis=(0, 1)).astype(int)
            avg_color_rgb = avg_color_bgr[::-1]  # Convert BGR to RGB

            # Find the closest texture
            block_texture_path = find_closest_texture(
                avg_color_rgb, block_paths, color_tree, block_data
            )

            # Load and resize the texture to fit the current block size
            texture = cv2.imread(block_texture_path)
            texture_resized = cv2.resize(texture, (block_width, block_height))

            # Replace the block with the resized texture
            output[y : y + block_height, x : x + block_width] = texture_resized

    return output


# Main Program
if __name__ == "__main__":
    input_image_path = "thing.jpg"  # Input image path
    csv_path = "block_data.csv"  # Path to the CSV file
    block_size = 32  # Make blocks larger

    # Load the input image
    input_image = cv2.imread(input_image_path)

    # Create the pixelated image
    pixelated = pixelated_image(input_image, block_size)

    # Display the pixelated image
    cv2.imwrite("pixelated_image.jpg", pixelated)
    cv2.imshow("Pixelated Image", pixelated)

    # Create the Minecraft mosaic
    minecraft_mosaic = pixelated_minecraft_mosaic(input_image, block_size, csv_path)

    # Save and display the Minecraft mosaic
    cv2.imwrite("minecraft_mosaic.jpg", minecraft_mosaic)
    cv2.imshow("Minecraft Mosaic", minecraft_mosaic)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
