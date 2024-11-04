from glob import glob
from typing import List, Optional

import cv2
import numpy as np
import rawpy
from astropy.stats import sigma_clip, sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.spatial import cKDTree

# Star detection constants
SIGMA_CLIP_SIGMA = 3.0
DAOSTAR_FINDER_FWHM = 4.0
DAOSTAR_FINDER_THRESHOLD_MULTIPLIER = 6.0  # Multiplier for standard deviation
MAX_ALIGNMENT_DISTANCE = 30  # Max distance for star matching in alignment (pixels)

# Image stacking constants
STACKING_SIGMA_CLIP = 3

# Mask constants
LANDSCAPE_MASK_RATIO = 0.9  # Fraction of image height to include in mask

# Paths and filenames
CR3_FILES_GLOB_PATTERN = "data/*.CR3"
OUTPUT_FILENAME = "stacked_image.jpg"


def save_ndarray_as_jpg(array: np.ndarray, filename: str = "output.jpg") -> None:
    # Normalize the array to the range 0-255
    normalized_array = cv2.normalize(
        src=array, dst=np.ndarray([]), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    # Convert to uint8
    uint8_array = normalized_array.astype(np.uint8)
    # Save the image
    cv2.imwrite(filename, uint8_array)


# Step 1: Load CR3 images (kept in color)
def load_cr3_images(cr3_files: List[str]) -> List[np.ndarray]:
    images = []
    for file in cr3_files:
        with rawpy.imread(file) as raw:
            rgb = raw.postprocess()
            images.append(rgb)
    return images


# Step 2: Convert images to grayscale for star detection
def convert_to_grayscale(images: List[np.ndarray]) -> List[np.ndarray]:
    gray_images = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images]
    return gray_images


# Step 3: Detect stars in each grayscale image
def detect_stars(
    images: List[np.ndarray], mask: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    star_coords_list = []
    for img in images:
        # Apply mask if provided
        if mask is not None:
            img_masked = img.copy()
            img_masked[~mask] = 0  # Zero out pixels outside the mask
        else:
            img_masked = img

        # Compute statistics over the unmasked area
        mean, median, std = sigma_clipped_stats(
            img_masked, mask=~mask if mask is not None else None, sigma=SIGMA_CLIP_SIGMA
        )

        daofind = DAOStarFinder(
            fwhm=DAOSTAR_FINDER_FWHM,
            threshold=DAOSTAR_FINDER_THRESHOLD_MULTIPLIER * std,
        )
        sources = daofind(img_masked - median)
        if sources is not None:
            print("Number of stars detected:", len(sources))
            positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
            star_coords_list.append(positions)
        else:
            print("No stars detected.")
            star_coords_list.append(np.array([]))
    return star_coords_list


# Step 4: Compute affine transformations based on star positions from grayscale images
def compute_affine_transforms(
    star_coords_list: List[np.ndarray],
) -> List[Optional[np.ndarray]]:
    reference_coords = star_coords_list[0]
    affine_transforms: List[Optional[np.ndarray]] = []

    # Build KDTree for the reference coordinates
    ref_tree = cKDTree(reference_coords)

    for i in range(1, len(star_coords_list)):
        coords = star_coords_list[i]

        if len(coords) == 0 or len(reference_coords) == 0:
            print(f"Skipping image {i} due to insufficient stars detected.")
            affine_transforms.append(None)
            continue

        # Find the nearest neighbors between the current and reference coordinates
        distances, indices = ref_tree.query(coords, k=1)

        # Define a maximum distance to consider a match valid (in pixels)
        mask = distances < MAX_ALIGNMENT_DISTANCE

        matched_src = coords[mask]
        matched_dst = reference_coords[indices[mask]]

        if len(matched_src) < 3:
            print(f"Not enough matched stars for image {i}. Skipping.")
            affine_transforms.append(None)
            continue

        # Compute affine transform using matched star positions
        M, inliers = cv2.estimateAffinePartial2D(matched_src, matched_dst)
        if M is not None:
            affine_transforms.append(M)
        else:
            print(f"Alignment failed for image {i}. Skipping.")
            affine_transforms.append(None)

    return affine_transforms


# Step 5: Apply affine transformations to color images
def apply_affine_transforms(
    images: List[np.ndarray], affine_transforms: List[Optional[np.ndarray]]
) -> List[np.ndarray]:
    reference_image = images[0]
    aligned_images = [reference_image]

    for i, M in enumerate(affine_transforms, start=1):
        img = images[i]

        if M is not None:
            # Apply warpAffine to each channel separately if needed
            aligned_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            aligned_images.append(aligned_img)
        else:
            print(f"Skipping application of affine transform for image {i}.")
            continue

    return aligned_images


# Step 6: Stack the aligned color images using sigma clipping
def stack_images(images: List[np.ndarray]) -> np.ndarray:
    # Convert list of images to a 4D numpy array: (num_images, height, width, channels)
    image_array = np.array(images)

    # Apply sigma clipping along the first axis (stack of images)
    clipped_images = sigma_clip(image_array, sigma=STACKING_SIGMA_CLIP, axis=0)

    # Compute the mean over the image stack (axis=0), preserving height, width, channels
    stacked_image = np.mean(clipped_images, axis=0)

    # Ensure the stacked image has the same data type as input images
    stacked_image = stacked_image.data  # Extract the data from the MaskedArray

    return stacked_image


# Step 7: Normalize and save the final stacked color image
def save_image(image: np.ndarray, filename: str = OUTPUT_FILENAME) -> None:
    # Normalize the image to 0-255 based on its data type
    if image.dtype == np.uint16 or image.max() > 255:
        # Normalize for 16-bit image
        max_value = np.max(image)
        image_norm = (image / max_value) * 255
    else:
        # Image is already in 8-bit range
        image_norm = image

    image_norm = image_norm.astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_norm, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_bgr)
    print(f"Stacked image saved as {filename}")


# Main function to execute the steps
def main() -> None:
    # Replace '*.cr3' with the path pattern to your CR3 files
    cr3_files: List[str] = glob(CR3_FILES_GLOB_PATTERN)
    if not cr3_files:
        print("No CR3 files found.")
        return

    print("Loading CR3 images...")
    raw_images: List[np.ndarray] = load_cr3_images(cr3_files)

    print("Converting images to grayscale...")
    gray_images: List[np.ndarray] = convert_to_grayscale(raw_images)

    print("Creating mask to exclude the landscape...")
    img_height, img_width = gray_images[0].shape
    mask: np.ndarray = np.zeros((img_height, img_width), dtype=bool)
    # Assuming the landscape is in the bottom portion of the image
    mask[0 : int(img_height * LANDSCAPE_MASK_RATIO), :] = True  # Adjust this as needed

    print("Detecting stars in each image...")
    star_coords_list: List[np.ndarray] = detect_stars(gray_images, mask=mask)

    print("Computing affine transformations based on star positions...")
    affine_transforms: List[Optional[np.ndarray]] = compute_affine_transforms(
        star_coords_list
    )

    print("Applying affine transformations to color images...")
    aligned_images: List[np.ndarray] = apply_affine_transforms(
        raw_images, affine_transforms
    )

    if len(aligned_images) < 2:
        print("Not enough images to stack after alignment.")
        return

    print("Stacking images...")
    stacked_image: np.ndarray = stack_images(aligned_images)

    print("Saving the final stacked image...")
    save_image(stacked_image, filename=OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
