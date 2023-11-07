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
from PIL import Image
import matplotlib.pyplot as plt
import requests
import json
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

def calculate_fov(tags, image_dimensions):
    # Extract focal length
    focal_length_tag = tags.get('EXIF FocalLength')
    focal_length = 0
    focal_length_35mm_tag = tags.get('EXIF FocalLengthIn35mmFilm')
    if focal_length_35mm_tag is not None and float(str(focal_length_35mm_tag)) > 0:
        focal_length = float(str(focal_length_35mm_tag))
    else:
        if focal_length_tag is not None:
            if "/" in str(focal_length_tag.values[0]):
                divisor, divident = str(focal_length_tag.values[0]).split("/")
                focal_length = float(str(divisor))/float(str(divident));
            else:
                focal_length = float(str(focal_length_tag.values[0]))
        else:
            focal_length = 4.15  # typical value for smartphone cameras in mm
    
    # Decide sensor size based on camera make and model
    make = str(tags.get('Image Make', ''))
    model = str(tags.get('Image Model', ''))
    
    # Decide if it's a smartphone camera or a regular one
    if 'iphone' in model.lower() or 'samsung' in model.lower() or 'nokia' in model.lower() or 'pixel' in model.lower():
        sensor_width = 6.16  # Assume sensor size for a typical smartphone camera in mm
        sensor_height = 4.62 # Typical sensor height for smartphones in mm
        print('phone camera')
    else:
        sensor_width = 35.9  # Assume sensor size for a full-frame camera in mm
        sensor_height = 24   # Typical sensor height for full-frame cameras in mm
        print('normal camera')
    # Check orientation and calculate FOV
    
    width, height = image_dimensions
    if height > width:  # Portrait orientation
        fov = 2 * math.degrees(math.atan2(sensor_height / 2, focal_length))
    else:  # Landscape orientation
        fov = 2 * math.degrees(math.atan2(sensor_width / 2, focal_length))

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
      
            # Convert Degrees Minutes Seconds to decimal degrees
            latitude_dd = dms_to_dd(latitude)
            longitude_dd = dms_to_dd(longitude)
            height_dd = height_to_dd(height)
            img = Image.open(file_path)
            dims = img.size         
            fov = calculate_fov(exif_data, dims)  # Pass sensor_width as None for now
            print(fov)
            # Move file to position subdirectory
            return latitude_dd, longitude_dd, height_dd, fov


def start_renderer(renderer_path, image_path, rotate_degrees, lat, long, height, fov):
    try:     
        parameters = f" {lat} {long} {height} {fov}"
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
            process.wait()        

    except Exception as e:
        print("Error:", str(e))
        return -1
        
        
def render_result(renderer_path, image_path, angle, lat, long, height, fov):
    try:
        parameters = f" {lat} {long} {height} {fov}"
        i = 0
        file_name, file_extension = os.path.splitext(os.path.basename(image_path))
        processes = []  # List to store subprocess objects

        orientation = f" {angle} 45"
        new_file_name = f"{file_name}_result_d{file_extension}"
        cmd = f"{renderer_path} {new_file_name} {parameters} {orientation}"
        print("Running:", cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
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
    best_match_image_degree = 0.0
    best_match_prob = 0.0
    best_match_angle = 0.0
    best_match_image_deg = 0
    best_match_h = None
    device = K.utils.get_cuda_device_if_available()
    if not os.path.exists('./matches'):
        os.makedirs('matches')
    while i <= 360:
        new_image_path = f"rendered_images/{file_name}_{i}_d{file_extension}"
        matched_image_path = f"matches/{file_name}_{i}_d_matched{file_extension}"


        img1 = load_torch_image(image_path, target_size)
        img2 = load_torch_image(new_image_path, target_size)

        matcher = KF.LoFTR(pretrained='outdoor')

        input_dict = {"image0": K.color.rgb_to_grayscale(img1), # LofTR works on grayscale images only 
                      "image1": K.color.rgb_to_grayscale(img2)}

        with torch.no_grad():
            correspondences = matcher(input_dict)
        mkpts0 = correspondences['keypoints0'].numpy()
        mkpts1 = correspondences['keypoints1'].numpy()
        if mkpts0.size < 10 or mkpts1.size < 10:
            i += rotate_degrees
            continue
        H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        inliers = inliers > 0
        if(inliers.size < 0):
            i += rotate_degrees
            continue
        if H is None or inliers is None:
            i += rotate_degrees
            continue
        
        # Normalize the homography matrix
        H = H / H[2, 2]
        
        # Calculate the rotation angle
        theta = np.arctan2(H[1, 0], H[0, 0])
        angle = np.degrees(theta)  # Convert radians to degrees
        match_prob = inliers.size
        # Update the best match
        if match_prob > best_match_prob or best_match_image_path == "":
            best_match_image_path = new_image_path
            best_match_angle = angle
            best_match_image_deg = i
            best_match_prob = match_prob
            best_match_h = H
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
        i += rotate_degrees
    # Print the results
    print(f"Best Match Image Path: {best_match_image_path}")
    print(f"Rotation Angle: {best_match_angle} degrees")
    if best_match_h is not None:
        # Load the best match image
        best_match_image = cv2.imread(best_match_image_path)
        # Load the original image
        original_image_color = cv2.imread(original_path)
        # Assuming the original image is not grayscale because we're going to overlay it
        h, w = original_image_color.shape[:2]
        # Apply the warpPerspective function with the correct parameters
        warped_image = cv2.warpPerspective(best_match_image, best_match_h, (w, h))

        # Overlay the warped image onto the original image
        # You can adjust the alpha value to make the overlay transparent
        alpha = 0.5
        overlay_image = cv2.addWeighted(original_image_color, 1 - alpha, warped_image, alpha, 0)

        # Save or show the overlay image
        overlay_image_path = f"overlay_{file_name}.png"
        cv2.imwrite(overlay_image_path, overlay_image)  # Saving the overlay image
        # cv2.imshow("Overlay Image", overlay_image)  # If you want to display the image
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()  # Make sure to destroy all windows if you've used cv2.imshow
    return best_match_angle + best_match_image_deg

if __name__ == "__main__":
    os.chdir('../build-peakfinder-Desktop_Qt_6_5_0_MinGW_64_bit-Release/plain_renderer')
    # Path to plain_renderer.exe
    renderer_path = "plain_renderer.exe "
    image_path = select_image()
    
    if not image_path:
        print("No image selected. Exiting.")
        exit()
    lat, long, height, fov = readExif(image_path)   
    rotate_degrees = round(fov - 10.0)    
    # Start the renderer with the specified parameters and select an image
    start_renderer(renderer_path,  image_path, rotate_degrees, lat, long, height, fov)
    
    angle = start_matching(image_path, rotate_degrees)
    
    render_result(renderer_path, image_path, angle, lat, long, height, fov)
    
    
