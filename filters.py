import cv2
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


# 2. Pencil Sketch Effect
def pencil_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def super_neon_glow_with_gradient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Reduce the thickness of the edges for fine lines
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Create a gradient of colors
    height, width = edges.shape
    gradient_colors = np.zeros((height, width, 3), dtype=np.float32)

    # Define gradient colors (HSV for smooth transitions)
    hsv_start = np.array([0, 255, 255])  # Red (HSV)
    hsv_end = np.array([120, 255, 255])  # Green (HSV)

    for y in range(height):
        ratio = y / height  # Vertical gradient
        hsv_color = hsv_start * (1 - ratio) + hsv_end * ratio
        rgb_color = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]
        gradient_colors[y, :] = rgb_color

    # Apply the gradient to the edges
    colored_edges = np.zeros_like(image, dtype=np.float32)
    for c in range(3):  # RGB channels
        colored_edges[:, :, c] = edges_dilated * gradient_colors[:, :, c] / 255

    # Add glow to the gradient lines
    glow = np.zeros_like(image, dtype=np.float32)
    for i in range(1, 6):  # Increase blur size for multi-layer glow
        blur_amount = i * 5
        if blur_amount % 2 == 0:
            blur_amount += 1
        blurred_layer = cv2.GaussianBlur(
            colored_edges, (blur_amount, blur_amount), sigmaX=0
        )
        glow += blurred_layer * (0.5 / i)  # Reduce intensity for further layers

    # Combine glow and original image
    final_glow = cv2.addWeighted(image.astype(np.float32), 0.5, glow, 0.8, 0)

    # Ensure pixel values are within [0, 255]
    final_glow = np.clip(final_glow, 0, 255).astype(np.uint8)
    return final_glow


# Minecraft


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
