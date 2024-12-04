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


# https://medium.com/dataseries/designing-image-filters-using-opencv-like-abode-photoshop-express-part-2-4479f99fb35
# 3. Comic Book Effect
def comic_effect(img):
    edges1 = cv2.bitwise_not(
        cv2.Canny(img, 100, 200)
    )  # for thin edges and inverting the mask obatined
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # applying median blur with kernel size of 5
    edges2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7
    )  # thick edges
    dst = cv2.edgePreservingFilter(
        img, flags=2, sigma_s=64, sigma_r=0.25
    )  # you can also use bilateral filter but that is slow
    # flag = 1 for RECURS_FILTER (Recursive Filtering) and 2 for  NORMCONV_FILTER (Normalized Convolution). NORMCONV_FILTER produces sharpening of the edges but is slower.
    # sigma_s controls the size of the neighborhood. Range 1 - 200
    # sigma_r controls the how dissimilar colors within the neighborhood will be averaged. A larger sigma_r results in large regions of constant color. Range 0 - 1
    cartoon1 = cv2.bitwise_and(
        dst, dst, mask=edges1
    )  # adding thin edges to smoothened image
    cartoon2 = cv2.bitwise_and(dst, dst, mask=edges2)

    return cartoon2


# 8. Neon Glow Effect
def neon_glow(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Dilate the edges to make them bolder
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Create a neon-colored overlay
    neon = np.zeros_like(image)
    neon[edges_dilated > 0] = [255, 0, 255]  # Neon pink

    # Blend the neon edges with the original image
    glow = cv2.addWeighted(image, 0.7, neon, 0.3, 0)
    return glow


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


# Main Program
if __name__ == "__main__":
    input_image = cv2.imread("sample.jpg")

    print("Apply glow gradient")
    neon_super_gradient = super_neon_glow_with_gradient(input_image)
    cv2.imwrite("filtered_photos/neon.jpg", neon_super_gradient)
    cv2.imshow("Super Neon Glow with Gradient Effect", neon_super_gradient)

    print("Apply pencil")
    sketch = pencil_sketch(input_image)
    cv2.imwrite("filtered_photos/pencil.jpg", sketch)
    cv2.imshow("Pencil Sketch Effect", sketch)

    print("Apply comic")
    comic = comic_effect(input_image)
    cv2.imwrite("filtered_photos/comic.jpg", comic)
    cv2.imshow("Comic Effect", comic)

    print("Apply Minecraft")
    pixelated = pixelated_image(input_image, 32)
    minecraft_mosaic = pixelated_minecraft_mosaic(input_image, 32, "block_data.csv")
    cv2.imwrite("filtered_photos/minecraft.jpg", minecraft_mosaic)
    cv2.imshow("Minecraft Block Effect", minecraft_mosaic)

    print("Finished")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
