#!/bin/bash

# Set input and output paths
input_image_before="/local_files/assignment/libya_floods_2023/processed/clipped_original_before.tif"
input_image_after="/local_files/assignment/libya_floods_2023/processed/clipped_original_after.tif"
output_folder="/local_files/assignment/libya_floods_2023/processed"

# Create output folder if it doesn't exist
mkdir -p $output_folder

# Shadow Detection using RGB bands on original images
otbcli_BandMath -il $input_image_before -out $output_folder/shadow_mask_before.tif -exp "im1b1 < 50 && im1b2 < 50 && im1b3 < 50 ? 1 : 0"
otbcli_BandMath -il $input_image_after -out $output_folder/shadow_mask_after.tif -exp "im1b1 < 50 && im1b2 < 50 && im1b3 < 50 ? 1 : 0"

# STEP 1: Normalize images
otbcli_BandMath -il $input_image_before -out $output_folder/norm_before.tif -exp "im1b1/255, im1b2/255, im1b3/255"
otbcli_BandMath -il $input_image_after -out $output_folder/norm_after.tif -exp "im1b1/255, im1b2/255, im1b3/255"

# Gaussian blur smoothing
otbcli_Smoothing -in $output_folder/norm_before.tif -out $output_folder/smooth_gaussian_before.tif -type gaussian
otbcli_Smoothing -in $output_folder/norm_after.tif -out $output_folder/smooth_gaussian_after.tif -type gaussian

# STEP 2: Segment images into raster files
otbcli_LargeScaleMeanShift -in $input_image_before -spatialr 5 -ranger 55 -minsize 300 -ram 1024 -mode raster -mode.raster.out $output_folder/segmentation_before.tif
otbcli_LargeScaleMeanShift -in $input_image_after -spatialr 5 -ranger 55 -minsize 300 -ram 1024 -mode raster -mode.raster.out $output_folder/segmentation_after.tif

# STEP 3: Classify images
otbcli_KMeansClassification -in $output_folder/smooth_gaussian_before.tif -ts 1000 -nc 10 -maxit 1000 -out $output_folder/kmeans_class_before.tif uint8
otbcli_KMeansClassification -in $output_folder/smooth_gaussian_after.tif -ts 1000 -nc 10 -maxit 1000 -out $output_folder/kmeans_class_after.tif uint8

# Apply shadow mask to KMeans classification
otbcli_BandMath -il $output_folder/kmeans_class_before.tif $output_folder/shadow_mask_before.tif -out $output_folder/kmeans_class_shadow_corrected_before.tif -exp "im2b1 == 1 ? 0 : im1b1"
otbcli_BandMath -il $output_folder/kmeans_class_after.tif $output_folder/shadow_mask_after.tif -out $output_folder/kmeans_class_shadow_corrected_after.tif -exp "im2b1 == 1 ? 0 : im1b1"

# STEP 4: Apply size filter using python script (calculate_segment.py)
python3 << EOF
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
segmentation_path_before = '$output_folder/segmentation_before.tif'
kmeans_path_before = '$output_folder/kmeans_class_shadow_corrected_before.tif'
output_path_before = '$output_folder/filtered_kmeans_class_before.tif'

segmentation_path_after = '$output_folder/segmentation_after.tif'
kmeans_path_after = '$output_folder/kmeans_class_shadow_corrected_after.tif'
output_path_after = '$output_folder/filtered_kmeans_class_after.tif'

# Define the size threshold (e.g., minimum size to filter out)
size_threshold = 2600  # Adjust this value based on your requirement

# Filter the segments for before and after images
filter_segments(segmentation_path_before, kmeans_path_before, output_path_before, size_threshold)
filter_segments(segmentation_path_after, kmeans_path_after, output_path_after, size_threshold)

EOF


# STEP 5: Reclassify the images
otbcli_BandMath -il $output_folder/filtered_kmeans_class_before.tif -out $output_folder/reclassified_kmeans_before.tif -exp "(im1b1>=0 && im1b1<5) ? 0 : ((im1b1>=5 && im1b1<=9) ? 1 : ((im1b1>9 && im1b1<=20) ? 0 : im1b1))"
otbcli_BandMath -il $output_folder/filtered_kmeans_class_after.tif -out $output_folder/reclassified_kmeans_after.tif -exp "(im1b1>=0 && im1b1<5) ? 0 : ((im1b1>=5 && im1b1<=9) ? 1 : ((im1b1>9 && im1b1<=20) ? 0 : im1b1))"

# SETP &: Remove noise from the images using opening and closing morphological operations
otbcli_BinaryMorphologicalOperation -in $output_folder/reclassified_kmeans_before.tif -out $output_folder/opened_kmeans_before.tif -filter opening -structype box -xradius 2 -yradius 2
otbcli_BinaryMorphologicalOperation -in $output_folder/reclassified_kmeans_after.tif -out $output_folder/opened_kmeans_after.tif -filter opening -structype box -xradius 2 -yradius 2
otbcli_BinaryMorphologicalOperation -in $output_folder/opened_kmeans_before.tif -out $output_folder/closed_kmeans_before.tif -filter closing -structype box -xradius 4 -yradius 4
otbcli_BinaryMorphologicalOperation -in $output_folder/opened_kmeans_after.tif -out $output_folder/closed_kmeans_after.tif -filter closing -structype box -xradius 4 -yradius 4

# STEP 7: Calculate the difference between the two images
otbcli_BandMath -il $output_folder/closed_kmeans_before.tif $output_folder/closed_kmeans_after.tif -out $output_folder/difference.tif -exp "im1b1-im2b1"

python3 << EOF
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
# Load the difference image
difference_image_path = "$output_folder/difference.tif"
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
EOF