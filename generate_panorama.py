import subprocess
import tkinter as tk
import exifread
import os
from tkinter import filedialog
import math
import cv2
import numpy as np
import kornia.feature as KF
import kornia as K
import torch
import torchvision.transforms as transforms
from kornia_moons.feature import *
import matplotlib.pyplot as plt
def load_torch_image(fname, target_size):
    # Load the image using OpenCV
    img = cv2.imread(fname)
    
    # Resize the image using OpenCV
    img = cv2.resize(img, target_size)
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img
    
    return img

def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an image file")
    return file_path

# Create a function to convert from Degrees Minutes Seconds to decimal degrees
def dms_to_dd(dms):
    degrees, minutes, seconds = [0,0,0]
    if "/" in str(dms.values[1]):
        divisor, divident = str(dms.values[1]).split("/")
        minutes = float(divisor) / float(divident)
    else:
        minutes = dms.values[1].num
        
    if "/" in str(dms.values[2]):
        divisor, divident = str(dms.values[2]).split("/")
        seconds = float(divisor) / float(divident)
    else:
        seconds = dms.values[2].num
        
    degrees = dms.values[0].num
    dd = float(degrees) + float(minutes) / 60 + float(seconds) / (60 * 60)
    return dd

def height_to_dd(height):
    height_dd = 0
    if "/" in str(height):
        divisor, divident = str(height).split("/")
        height_dd = float(divisor) / float(divident)
    else: 
        height_dd = height
    return height_dd

def calculate_fov(focal, lenswidth):
    fov = (2 * math.atan(lenswidth / (2 * focal))) * 180 / math.pi
    return fov


def readExif(file_path):
    with open(file_path, 'rb') as f:
        # Read EXIF data
        exif_data = exifread.process_file(f)
        # Check if GPS info is present
        if not exif_data.get('GPS GPSLatitude'):
            # Move file to noposition subdirectory
            f.close()
            move(file_path, noposition_dir + "/" + file_name)
        else:
            # Get GPS data
            latitude = exif_data.get('GPS GPSLatitude')
            longitude = exif_data.get('GPS GPSLongitude')
            height = exif_data.get('GPS GPSAltitude')
            focal_length = exif_data.get('EXIF FocalLength')
            camera_model = exif_data.get('Image Model')
            if "/" in str(focal_length.values[0]):
                divisor, divident = str(focal_length.values[0]).split("/")
                focal_length = float(str(divisor))/float(str(divident));
            else:
                focal_length = float(str(focal_length))
            # Convert Degrees Minutes Seconds to decimal degrees
            latitude_dd = dms_to_dd(latitude)
            longitude_dd = dms_to_dd(longitude)
            height_dd = height_to_dd(height)

            # Calculate FOV
            lenswidth = 35.9
            if camera_model == "SONY":
                lenswidth = 35.9
            elif camera_model == "HMD Global":
                lenswidth = 5.839
            fov = calculate_fov(focal_length, lenswidth)  # Pass sensor_width as None for now
            fov = 70
            # Move file to position subdirectory
            return f" {latitude_dd} {longitude_dd} {height_dd} {fov}"


def start_renderer(renderer_path, image_path, rotate_degrees):
    try:
        parameters = readExif(image_path)
        orientation = 25.0
        i = 0
        file_name, file_extension = os.path.splitext(os.path.basename(image_path))
        processes = []  # List to store subprocess objects

        while i <= 360:
            orientation = f" {i} 45"
            new_file_name = f"{file_name}_{i}_d{file_extension}"
            cmd = f"{renderer_path} {new_file_name} {parameters} {orientation}"
            i += rotate_degrees
            print("Running:", cmd)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            processes.append(process)

        # Wait for all subprocesses to finish
        for process in processes:
            process.wait()

    except Exception as e:
        print("Error:", str(e))
        return -1
        
def start_matching(image_path, rotate_degrees):
    i = 0
    target_size = (512, 512)
    original_path = image_path
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load the original image in grayscale
    best_match_image_path = ""
    best_match_prob = 0.0
    device = K.utils.get_cuda_device_if_available()
    while i <= 360:
        new_image_path = f"D:/AlpineMaps/images/single_render/{file_name}_{i}_d{file_extension}"
        matched_image_path = f"D:/AlpineMaps/images/matched/{file_name}_{i}_d_matched{file_extension}"
        i += rotate_degrees

        img1 = load_torch_image(image_path, target_size)
        img2 = load_torch_image(new_image_path, target_size)

        matcher = KF.LoFTR(pretrained='outdoor')

        input_dict = {"image0": K.color.rgb_to_grayscale(img1), # LofTR works on grayscale images only 
                      "image1": K.color.rgb_to_grayscale(img2)}

        with torch.no_grad():
            correspondences = matcher(input_dict)
        mkpts0 = correspondences['keypoints0'].numpy()
        mkpts1 = correspondences['keypoints1'].numpy()
        H, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        inliers = inliers > 0
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
        torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
        K.tensor_to_image(img1),
        K.tensor_to_image(img2),
        inliers,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                   'tentative_color': None, 
                   'feature_color': (0.2, 0.5, 1), 'vertical': False},
                   ax=ax,)
        plt.savefig(matched_image_path)

if __name__ == "__main__":
    # Path to plain_renderer.exe
    renderer_path = "E:/Github/Alpinemaps/build-peakfinder-Desktop_Qt_6_5_0_MinGW_64_bit-Release/plain_renderer/plain_renderer.exe "

    image_path = select_image()
    rotate_degrees = 45
    if not image_path:
        print("No image selected. Exiting.")
        exit()
                        
    # Start the renderer with the specified parameters and select an image
    start_renderer(renderer_path,  image_path, rotate_degrees)
    
    start_matching(image_path, rotate_degrees)
    
    
