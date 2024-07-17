# AOHRSI
ANALYSIS OF HIGH-RESOLUTION REMOTE SENSING IMAGERY

## Script for Libya Flood Analysis

### Requirements
- Docker image with OTB (Orfeo ToolBox) installed
- Libraries: numpy, gdal
- Input data: Pre- and post-flood satellite images with RGB bands

### Step 1: Install Required Libraries and Tools

TODO

### Step 2: Normalize Images
**Explanation:**
- **Why:** Normalization scales the pixel values of the images to a common range (0 to 1 in this case).
- **How:** This is done by dividing each band by 255, the maximum value for an 8-bit image.
- **Objective:** Standardizing pixel values helps in reducing variability due to lighting conditions and other factors, making subsequent analysis more robust and accurate.

```shell
otbcli_BandMath -il /content/clipped_original_before.tif -out /content/norm_before.tif -exp "im1b1/255, im1b2/255, im1b3/255"
otbcli_BandMath -il /content/clipped_original_after.tif -out /content/norm_after.tif -exp "im1b1/255, im1b2/255, im1b3/255"
```

### Step 3: Apply Gaussian Blur Smoothing
**Explanation:**
- **Why:** Smoothing with a Gaussian blur reduces noise and small variations in the image.
- **How:** Gaussian blur applies a convolution with a Gaussian function, effectively averaging nearby pixel values.
- **Objective:** Reducing noise helps in more accurate segmentation and classification in later steps.

```shell
otbcli_Smoothing -in /content/norm_before.tif -out /content/smooth_gaussian_before.tif -type gaussian
otbcli_Smoothing -in /content/norm_after.tif -out /content/smooth_gaussian_after.tif -type gaussian
```

### Step 4: Shadow Detection
**Explanation:**
- **Why:** Detect shadows to avoid misinterpreting them as buildings.
- **How:** Use a threshold on the RGB bands to identify dark areas as shadows.
- **Objective:** Create a mask to exclude shadowed areas from further analysis.

```shell
otbcli_BandMath -il /content/clipped_original_before.tif -out /content/shadow_mask_before.tif -exp "im1b1 < 50 && im1b2 < 50 && im1b3 < 50 ? 1 : 0"
otbcli_BandMath -il /content/clipped_original_after.tif -out /content/shadow_mask_after.tif -exp "im1b1 < 50 && im1b2 < 50 && im1b3 < 50 ? 1 : 0"
```

### Step 5: Apply Shadow Mask
**Explanation:**
- **Why:** Ensure that shadowed areas are not classified as buildings.
- **How:** Use the shadow mask to set the values in the KMeans classification to 0 where shadows are detected.
- **Objective:** Correct the KMeans classification results by removing shadowed areas.

```shell
otbcli_BandMath -il /content/kmeans_class_before.tif /content/shadow_mask_before.tif -out /content/kmeans_class_shadow_corrected_before.tif -exp "im2b1 == 1 ? 0 : im1b1"
otbcli_BandMath -il /content/kmeans_class_after.tif /content/shadow_mask_after.tif -out /content/kmeans_class_shadow_corrected_after.tif -exp "im2b1 == 1 ? 0 : im1b1"
```

### Step 6: Segment Images
**Explanation:**
- **Why:** Image segmentation partitions the image into segments or regions based on pixel similarity.
- **How:** The Large Scale Mean Shift algorithm groups pixels into segments with similar color and spatial properties.
- **Objective:** Segmentation helps in isolating meaningful regions in the image, such as areas affected by the flood.

```shell
otbcli_LargeScaleMeanShift -in /content/clipped_original_before.tif -spatialr 5 -ranger 55 -minsize 300 -ram 1024 -mode raster -mode.raster.out /content/segmentation_before.tif
otbcli_LargeScaleMeanShift -in /content/clipped_original_after.tif -spatialr 5 -ranger 55 -minsize 300 -ram 1024 -mode raster -mode.raster.out /content/segmentation_after.tif
```

### Step 7: Classify Images using KMeans
**Explanation:**
- **Why:** Classification groups pixels into clusters based on their spectral properties.
- **How:** KMeans clustering assigns each pixel to one of the specified number of clusters (10 in this case).
- **Objective:** Classifying the image helps in identifying different land cover types or features within the image, which can be compared before and after the flood.

```shell
otbcli_KMeansClassification -in /content/smooth_gaussian_before.tif -ts 1000 -nc 10 -maxit 1000 -out /content/kmeans_class_before.tif uint8
otbcli_KMeansClassification -in /content/smooth_gaussian_after.tif -ts 1000 -nc 10 -maxit 1000 -out /content/kmeans_class_after.tif uint8
```

### Step 8: Filter Segments by Size
**Explanation:**
- **Why:** Filtering segments based on size removes small, insignificant regions that might be noise or irrelevant details.
- **How:** The Python script calculates the size of each segment and masks out segments larger than a specified threshold.
- **Objective:** Removing large segments focuses the analysis on smaller, potentially more relevant changes in the image.

```python
import numpy as np
from osgeo import gdal

# Enable GDAL exceptions
gdal.UseExceptions()

# Function to filter segments by size
def filter_segments(segmentation_path, kmeans_path, output_path, size_threshold):
    try:
        # Load the rasterized segmentation
        segmentation_raster = gdal.Open(segmentation_path)
        segmentation_band = segmentation_raster.GetRasterBand(1)
        segmentation = segmentation_band.ReadAsArray()

        # Load the KMeans classified image
        kmeans_raster = gdal.Open(kmeans_path)
        kmeans_band = kmeans_raster.GetRasterBand(1)
        kmeans_classified = kmeans_band.ReadAsArray()

        # Calculate the size of each segment
        unique, counts = np.unique(segmentation, return_counts=True)
        segment_sizes = dict(zip(unique, counts))

        # Create a mask where segments larger than the threshold are set to 0
        mask = np.isin(segmentation, [seg for seg, size in segment_sizes.items() if size >= size_threshold])
        filtered_kmeans_classified = np.where(mask, 0, kmeans_classified)

        # Save the filtered KMeans classified image as a new raster
        driver = gdal.GetDriverByName('GTiff')
        out_raster = driver.Create(output_path, kmeans_raster.RasterXSize, kmeans_raster.RasterYSize, 1, gdal.GDT_Byte)
        out_band = out_raster.GetRasterBand(1)
        out_band.WriteArray(filtered_kmeans_classified)

        # Copy georeference information
        out_raster.SetGeoTransform(kmeans_raster.GetGeoTransform())
        out_raster.SetProjection(kmeans_raster.GetProjection())

        out_band.FlushCache()
        out_raster = None

    except Exception as e:
        print(f"An error occurred: {e}")

# Paths to input and output files
segmentation_path_before = '/content/segmentation_before.tif'
kmeans_path_before = '/content/kmeans_class_shadow_corrected_before.tif'
output_path_before = '/content/filtered_kmeans_class_before.tif'

segmentation_path_after = '/content/segmentation_after.tif'
kmeans_path_after = '/content/kmeans_class_shadow_corrected_after.tif'
output_path_after = '/content/filtered_kmeans_class_after.tif'

# Define the size threshold (e.g., minimum size to filter out)
size_threshold = 2600  # Adjust this value based on your requirement

# Filter the segments for before and after images
filter_segments(segmentation_path_before, kmeans_path_before, output_path_before, size_threshold)
filter_segments(segmentation_path_after, kmeans_path_after, output_path_after, size_threshold)
```

### Step 9: Reclassify Images
**Explanation:**
- **Why:** Reclassification consolidates or reassigns classes based on specific criteria to highlight certain features.
- **How:** Using BandMath expressions, pixels are reassigned to new classes based on their original classification.
- **Objective:** This step simplifies the classification, making it easier to analyze specific changes of interest.

```shell
otbcli_BandMath -il /content/filtered_kmeans_class_before.tif -out /content/reclassified_kmeans_before.tif -exp "(im1b1>=0 && im1b1<5) ? 0 : ((im1b1>=5 && im1b1<=9) ? 1 : ((im1b1>9 && im1b1<=20) ? 0 : im1b1))"
otbcli_BandMath -il /content/filtered_kmeans_class_after.tif -out /content/reclassified_kmeans_after.tif -exp "(im1b1>=0 && im1b1<5) ? 0 : ((im1b1>=5 && im1b1<=9) ? 1 : ((im1b1>9 && im1b1<=20) ? 0 : im1b1))"
```

### Step 10: Remove Noise Using Morphological Operations
**Explanation:**
- **Why:** Morphological operations like opening and closing help in removing small noise and smoothing the boundaries of segments.
- **How:** Opening removes small objects from the foreground (useful for noise removal), and closing fills small holes in the foreground.
- **Objective:** These operations further clean the images, making the detected changes more reliable.

```shell
otbcli_BinaryMorphologicalOperation -in /content/reclassified_kmeans_before.tif -out /content/opened_kmeans_before.tif -filter opening -structype box -xradius 1 -yradius 1
otbcli_BinaryMorphologicalOperation -in /content/reclassified_kmeans_after.tif -out /content/opened_kmeans_after.tif -filter opening -structype box -xradius 2 -yradius 2
otbcli_BinaryMorphologicalOperation -in /content/opened_kmeans_before.tif -out /content/closed_kmeans_before.tif -filter closing -structype box -xradius 2 -yradius 2
otbcli_BinaryMorphologicalOperation -in /content/opened_kmeans_after.tif -out /content/closed_kmeans_after.tif -filter closing -structype box -xradius 4 -yradius 4
```

### Step 11: Calculate the Difference Between Images
**Explanation:**
- **Why:** Calculating the difference between the before and after images highlights areas of change.
- **How:** The difference is computed using pixel-wise subtraction of the "before" image from the "after" image.
- **Objective:** This step directly quantifies changes, helping to identify areas impacted by the flood.

```shell
otbcli_BandMath -il /content/closed_kmeans_before.tif /content/closed_kmeans_after.tif -out /content/difference.tif -exp "im1b1-im2b1"
```

### Step 12: Analyze the Difference Image
**Explanation:**
- **Why:** Analyzing the difference image helps in quantifying the extent and type of changes.
- **How:** Using Python, calculate the number of pixels representing no change, positive change, and negative change, then determine their percentages.
- **Objective:** This analysis provides a quantitative measure of the flood's impact, which is essential for damage assessment and recovery planning.

```python
import numpy as np
from osgeo import gdal

gdal.UseExceptions()
# Load the difference image
difference_image_path = "/content/difference.tif"
difference_dataset = gdal.Open(difference_image_path)
difference_band = difference_dataset.GetRasterBand(1)
difference_array = difference_band.ReadAsArray()

# Define the thresholds
no_change_threshold = 0
positive_change_threshold = 0  # Change this value if you consider only significant positive changes
negative_change_threshold = 0  # Change this value if you consider only significant negative changes

# Calculate the total number of pixels
total_pixels = difference_array.size

# Calculate the number of pixels that haven't changed
no_change_pixels = np.sum(difference_array == no_change_threshold)

# Calculate the number of pixels with positive changes
positive_change_pixels = np.sum(difference_array > positive_change_threshold)

# Calculate the number of pixels with negative changes
negative_change_pixels = np.sum(difference_array < negative_change_threshold)

# Calculate percentages
no_change_percentage = (no_change_pixels / total_pixels) * 100
positive_change_percentage = (positive_change_pixels / total_pixels) * 100
negative_change_percentage = (negative_change_pixels / total_pixels) * 100

# Print the results
print(f"No change percentage: {no_change_percentage:.2f}%")
print(f"Positive change percentage: {positive_change_percentage:.2f}%")
print(f"Negative change percentage: {negative_change_percentage:.2f}%")
```