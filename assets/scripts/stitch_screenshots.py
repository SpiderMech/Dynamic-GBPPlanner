#!/usr/bin/env python3
"""
Script to stitch screenshots from ./assets/screenshots into a horizontal grid of 3 images.
Adds black borders around each image for clear separation.
"""

import os
import glob
from PIL import Image
import numpy as np

def stitch_images():
    # Get all image files from screenshots directory
    screenshot_dir = "./assets/screenshots"
    image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(screenshot_dir, pattern)))
    
    # Sort the files to ensure consistent ordering
    image_files.sort()
    
    if not image_files:
        print("No images found in ./assets/screenshots")
        return
    
    print(f"Found {len(image_files)} images:")
    for img_file in image_files:
        print(f"  - {os.path.basename(img_file)}")
    
    # Load all images
    images = []
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
            print(f"Loaded: {os.path.basename(img_path)} - Size: {img.size}")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return
    
    if not images:
        print("No valid images loaded")
        return
    
    # Border settings
    border_width = 5  # Black border width in pixels
    
    # Get dimensions - find the maximum height and width to ensure uniformity
    max_height = max(img.height for img in images)
    max_width = max(img.width for img in images)
    
    # Calculate total dimensions including borders
    # Each image gets bordered on all sides, plus gaps between images
    bordered_width = max_width + 2 * border_width
    bordered_height = max_height + 2 * border_width
    total_width = len(images) * bordered_width
    
    print(f"Individual image max size: {max_width} x {max_height}")
    print(f"Bordered image size: {bordered_width} x {bordered_height}")
    print(f"Final grid dimensions: {total_width} x {bordered_height}")
    
    # Create the stitched image with black background
    stitched = Image.new('RGB', (total_width, bordered_height), color='black')
    
    # Paste images horizontally with borders
    x_offset = 0
    for i, img in enumerate(images):
        # Center the image within its bordered cell
        img_x = x_offset + border_width + (max_width - img.width) // 2
        img_y = border_width + (max_height - img.height) // 2
        
        stitched.paste(img, (img_x, img_y))
        x_offset += bordered_width
        print(f"Placed image {i+1} at position ({img_x}, {img_y})")
    
    # Save the result
    output_path = "./stitched_screenshots.png"
    stitched.save(output_path, quality=95)
    print(f"\nStitched image saved as: {output_path}")
    print(f"Final dimensions: {stitched.size}")

if __name__ == "__main__":
    stitch_images()
