import cv2
import numpy as np
import os

####################### Image transformations #######################
def horizontal_flip(image: np.ndarray, output_dir: str) -> np.ndarray:
    flipped_image = np.flip(image, axis=1)
    cv2.imwrite(os.path.join(output_dir, "horizontal_flip.png"), flipped_image)
    return flipped_image


def scale_image(image: np.ndarray, scale_factor: float, output_dir:str) -> np.ndarray:
    h, w = image.shape[:2] #height & width
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    # Resize to new scaled size
    scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    start_X = (new_w - 112) // 2
    start_Y = (new_h -112) // 2

    cropped_img = scaled_image[start_Y : start_Y + 112, start_X : start_X + 112]

    cv2.imwrite(os.path.join(output_dir, "scaled_image.png"), cropped_img)
    return cropped_img

#Check range 
def gaussian_blur(image: np.ndarray, output_dir: str, sigma_range: tuple = (0.1, 0.8)) -> np.ndarray:
    sigma = np.random.uniform(*sigma_range)
    blurred_img = cv2.GaussianBlur(image, (0, 0), sigma) #opencv geenrates a kernel size
    cv2.imwrite(os.path.join(output_dir, "gaussianBlur.png"), blurred_img)
    print(f"Saved blurred image to: {output_dir} with sigma = {sigma:.3f}")
    return blurred_img

def adjust_brightness(image: np.ndarray, output_dir, factor_range: tuple = (0.6, 1.4)) -> np.ndarray:
    factor = np.random.uniform(*factor_range)
    image_brightness = np.clip(image * factor, 0, 255).astype(image.dtype)
    cv2.imwrite(os.path.join(output_dir, "adjBrightness.png"), image_brightness)
    print (f"Saved image with factor{factor:.3f}")
    return image_brightness

def adjust_contrast(image: np.ndarray, output_dir, factor_range=(0.75, 1.25)):
    factor = np.random.uniform(*factor_range)
    mean = image.mean()
    adjContrast = np.clip((image - mean) * factor + mean, 0, 255).astype(image.dtype)
    cv2.imwrite(os.path.join(output_dir, "adjContrast.png"), adjContrast)
    print (f"Saved image with factor{factor:.3f}")
    return adjContrast


####################### Coordinate transformations #######################

def transform_coords(x, y, original_size, scale_factor, final_size=(112, 112)):
    orig_h, orig_w = original_size
    new_h = int(orig_h * scale_factor)
    new_w = int(orig_w * scale_factor)

    # Step 1: Scale the coordinates
    x_scaled = x * scale_factor
    y_scaled = y * scale_factor

    # Step 2: Resize into 112x112 space
    # Coordinates must also be scaled according to how the image was resized to 112x112
    scale_x = final_size[1] / new_w
    scale_y = final_size[0] / new_h

    x_final = x_scaled * scale_x
    y_final = y_scaled * scale_y

    return x_final, y_final

def horizontal_coordinate_flip(coordinates: np.ndarray, image_width: int) -> np.ndarray:
    flipped_coords = coordinates.copy()
    flipped_coords[:, 0] = image_width - flipped_coords[:, 0]
    return flipped_coords

###################### Main function######################
def main():
    image = cv2.imread("/vol/bitbucket/nc624/sam2/outputs/videoFrame/0X105B9EF57DE45DCB/00121.jpg")
    output_dir = "/vol/bitbucket/nc624/sam2/Test_files/augmented_images"
    os.makedirs(output_dir, exist_ok=True)

    #horizontal_flip(image, output_dir)
    #scale_image(image, 1.3, output_dir)
    #gaussian_blur(image, output_dir)
    #adjust_brightness(image, output_dir)
    adjust_contrast(image, output_dir)

print("Image saved!")

if __name__ == "__main__":
    main()