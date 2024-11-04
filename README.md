# Image Stacking Script for Astrophotography

## Description

This script automates the process of stacking multiple astrophotography images to improve the signal-to-noise ratio and bring out faint celestial objects. It performs the following steps:

1. **Load Canon CR3 Raw Images**: Reads raw images from a specified directory.
2. **Convert to Grayscale**: Prepares images for star detection while preserving color data for the final output.
3. **Create Mask**: Excludes unwanted parts of the image (e.g., the landscape) from star detection.
4. **Detect Stars**: Uses the DAOStarFinder algorithm to identify stars in each image.
5. **Compute Affine Transformations**: Aligns images based on the detected star positions.
6. **Apply Transformations**: Warps the original color images to align them perfectly.
7. **Stack Images**: Combines the aligned images using sigma clipping to reduce noise.
8. **Save Final Image**: Outputs the stacked image in JPEG (8bit) or TIFF (16bit) format.

## Features

- **Automatic Star Detection and Alignment**: Uses star detection algorithms to align images based on star positions.
- **Sigma-Clipped Stacking**: Reduces noise and improves image quality by stacking images with sigma clipping.
- **Landscape Masking**: Excludes the landscape portion of the images to focus on the sky.

## Requirements

- **Python 3.12** (possibly works with other versions but not tested)

### System Packages
- **python3.12-venv** (for creating virtual environments, install with `sudo apt install python3.12-venv`)

### Python Packages

- `opencv-python`
- `numpy`
- `rawpy`
- `astropy`
- `photutils`
- `scipy`

## Installation

1. **Clone the Repository or Download the Script**

   ```bash
   git clone https://github.com/fduris/night_sky_stacking.git
   ```

2. **Install Dependencies**

   ```bash
   cd /path/to/project
   sudo apt install python3.12-venv
   python3 -m venv venv
   source venv/bin/activate
   pip install .
   ```

## Usage

1. **Adjust Script Parameters (Optional)**

   Open the script and modify the constants at the top if necessary:

   - **Star Detection Constants**

     - `STAR_DETECTION_SIGMA_CLIP`: Default is `3.0`. Affects how aggressively outliers are excluded.
     - `DAOSTAR_FINDER_FWHM`: Default is `4.0`. Approximate full width at half maximum (FWHM) of the stars in the image.
     - `DAOSTAR_FINDER_THRESHOLD_MULTIPLIER`: Default is `6.0`. Multiplier for the standard deviation to set the detection threshold.

   - **Mask Constants**

     - `LANDSCAPE_MASK_RATIO`: Default is `0.9`. Fraction of the image height to include in the mask for star detection. This depends on how big part a landscape takes in your images. Since we want to align the images based on the moving stars, removing stationary landscape helps. The value is interpreted as a percentage of the image that is processed taken from the top. The landscape will still be visible in the stacked image but it will be fuzzy.

   - **Alignment Constants**

     - `MAX_ALIGNMENT_DISTANCE`: Default is `30` pixels. Maximum pixel distance to consider when matching stars between images. This depends how far you expect your stars to "travel" in the images which depends on how many shots you have and how long the exposure time is.

   - **Image Stacking Constants**

     - `STACKING_SIGMA_CLIP`: Default is `3`.

   - **Paths and Filenames**

     - `CR3_FILES_GLOB_PATTERN`: Default is `"data/*.CR3"`. Glob pattern to locate CR3 files.
     - `OUTPUT_FILENAME`: Default is `"stacked_image.jpg"`. Filename for the final stacked image.

2. **Run the Script**

   ```bash
   cd /path/to/project
   source venv/bin/activate
   python main.py
   ```

3. **Output**

   The final stacked image will be saved in the project root directory with the name specified in `OUTPUT_FILENAME` with file extension based on output type (tiff or jpg).

## How It Works

1. **Loading Images**

   - The script searches for CR3 files using the glob pattern specified.
   - Raw images are loaded and converted to RGB format using `rawpy`.

2. **Grayscale Conversion**

   - Images are converted to grayscale using OpenCV for star detection purposes.

3. **Creating the Mask**

   - A binary mask is created to exclude the lower portion of the images (assumed to be the landscape).
   - The mask is applied during star detection to focus on the sky area.

4. **Star Detection**

   - Uses `sigma_clipped_stats` to calculate the background statistics.
   - `DAOStarFinder` identifies stars based on the calculated statistics.
   - Detected star positions are stored for each image.

5. **Computing Affine Transformations**

   - The first image is used as the reference.
   - For each subsequent image, stars are matched to the reference using a KDTree.
   - Affine transformations are computed to align the images based on matched stars.

6. **Applying Transformations**

   - Affine transformations are applied to the original color images.
   - Aligned images are added to a list for stacking.

7. **Stacking Images**

   - Images are stacked using `sigma_clip` to reduce noise.
   - The mean of the sigma-clipped images is computed to create the final image.

8. **Saving the Final Image**

   - The stacked image is normalized and converted to 8-bit format.
   - Saved as a JPEG file using OpenCV.

## Troubleshooting

- **No CR3 Files Found**

  - Ensure that your images are in the correct directory.
  - Verify that `CR3_FILES_GLOB_PATTERN` matches the location of your images.

- **No Stars Detected**

  - Adjust `DAOSTAR_FINDER_FWHM` and `DAOSTAR_FINDER_THRESHOLD_MULTIPLIER` star detection parameters.
  - Check if the mask `LANDSCAPE_MASK_RATIO` is excluding too much or too little of the image.

- **Alignment Issues**

  - Ensure that images have overlapping star fields.
  - Adjust `MAX_ALIGNMENT_DISTANCE` if stars are not matching properly.

- **Final Image Not as Expected**

  - Experiment with different stacking sigma values in `STACKING_SIGMA_CLIP`.
  - You can also try to change arguments of `raw.postprocess()` in `load_cr3_images()` function to get better results. See [rawpy documentation](https://letmaik.github.io/rawpy/api/rawpy.Params.html).

## Example

You can use example CR3 images in the `data/` directory with the given defaults to produce a stacked image.

## License

This project is licensed under the [GPLv3 License](LICENSE).

## Acknowledgments

- Utilizes the [OpenCV](https://opencv.org/) library for image processing.
- Star detection powered by [Photutils](https://photutils.readthedocs.io/en/stable/).
- Raw image processing with [rawpy](https://letmaik.github.io/rawpy/api/rawpy.RawPy.html).

## Disclaimer

This script is provided "as is" without warranty of any kind. Use it at your own risk.
