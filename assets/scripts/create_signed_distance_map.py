import os
import numpy as np
from argparse import ArgumentParser
import cv2
from scipy.ndimage import distance_transform_edt, sobel
import matplotlib.pyplot as plt

def pre_process(fp="../imgs/magic_roundabout/magic_roundabout_pre.png"):
    """Read and process image"""
    im = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(im, 127, 1, cv2.THRESH_BINARY)
    return binary_image

def signed_distance_map(binary_image):
    """Compute signed distance map (negative inside objects, positive outside)"""
    ext_dist_map = distance_transform_edt(binary_image)
    int_dist_map = distance_transform_edt(1- binary_image)
    return ext_dist_map - int_dist_map

def calculate_jacobian(signed_dist):
    """Returns the Jacobians in X and Y directions"""
    dy, dx = np.gradient(signed_dist)
    return dx, dy

def draw(sdf, jac_x, jac_y):
    """Visualise the SDF and Jacobians"""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    im0 = ax[0].imshow(sdf, cmap='jet')
    ax[0].set_title('Signed Distance Map')
    plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(jac_x, cmap='jet')
    ax[1].set_title('Jacobian (X component)')
    plt.colorbar(im1, ax=ax[1])

    im2 = ax[2].imshow(jac_y, cmap='jet')
    ax[2].set_title('Jacobian (Y component)')
    plt.colorbar(im2, ax=ax[2])
    plt.show()

def save(output_name, sdf, jac_x, jac_y):
    """Save as npy float32 files"""
    np.save(f"{output_name}_sdf.npy", sdf.astype(np.float32))
    np.save(f"{output_name}_jac_X.npy", jac_x.astype(np.float32))
    np.save(f"{output_name}_jac_Y.npy", jac_y.astype(np.float32))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", default=None)
    args = parser.parse_args()
    input_fp = args.input
    if ((input_fp is None) or (not os.path.isfile(input_fp))):
        raise FileNotFoundError("Invalid input file path given.")
    output_name = input_fp.split("/")[-1].split(".")[0]

    bin_im = pre_process(input_fp)
    sdf = signed_distance_map(bin_im)
    jac_x, jac_y = calculate_jacobian(sdf)
    draw(sdf, jac_x, jac_y)
    save(output_name, sdf, jac_x, jac_y)

