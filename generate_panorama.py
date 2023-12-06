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
from database import *

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
    
def change_extension_to_txt(file_path, base_name_replace):
    """
    Changes the file extension to .txt and replaces the base name.
    
    :param file_path: Path of the original file
    :param base_name_replace: Tuple containing (string_to_replace, new_string)
    :return: Path with the new extension and base name
    """
    directory, filename = os.path.split(file_path)
    name, _ = os.path.splitext(filename)  # Splitting off the extension
    new_name = name.replace(*base_name_replace) + '.txt'  # Replacing base name and adding new extension
    return os.path.join(directory, new_name)   

def yaw_to_compass(yaw_radians):
    """
    Converts yaw from radians to degrees with north as 0 degrees.

    :param yaw_radians: Yaw in radians, where 0 radians is west
    :return: Yaw in degrees with north as 0 degrees
    """
    # Convert from radians to degrees
    yaw_degrees = yaw_radians * (180 / math.pi)

    yaw_degrees_adjusted = yaw_degrees + 90

    # Normalize to 0 - 360 degrees range
    yaw_degrees_normalized = yaw_degrees_adjusted % 360

    return yaw_degrees_normalized
   
def read_info(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # Assuming the file structure is consistent and correct
            fov = float(lines[5].strip())
            height = float(lines[4].strip())
            original_latitude = float(lines[2].strip())
            original_longitude = float(lines[3].strip())
            print(yaw_to_compass(float(lines[1].strip().split()[0])))
            return original_latitude, original_longitude,height,  fov
    except Exception as e:
        print("Error:", str(e))
        return None, None, None, None

def scale_to_fit_screen(image_width, image_height, screen_width, screen_height):
    # Calculate the aspect ratio of both the image and the screen
    image_aspect_ratio = image_width / image_height
    screen_aspect_ratio = screen_width / screen_height
    ratio = 0
    # Determine if the image needs to be scaled based on width or height
    if image_aspect_ratio > screen_aspect_ratio:
        # Scale based on width
        scaled_width = screen_width
        scaled_height = scaled_width / image_aspect_ratio
    else:
        # Scale based on height
        scaled_height = screen_height
        scaled_width = scaled_height * image_aspect_ratio
    
    # Ensure that the scaled dimensions are integers and do not exceed the screen size
    scaled_width = min(screen_width, int(scaled_width))
    scaled_height = min(screen_height, int(scaled_height))
    
    return scaled_width, scaled_height
   


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
        fov = 2 * math.degrees(math.atan2(35.9 / 2, focal_length))
        print("using 35mm focal length")
        return fov
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


def start_renderer(renderer_path, image_path, rotate_degrees, lat, long, alt, fov):
    try: 
        img = Image.open(image_path)
        width, height = img.size
        width, height = scale_to_fit_screen(width, height, 1920, 1080)        
        parameters = f" {lat} {long} {alt} {fov}"
        i = 0
        file_name, file_extension = os.path.splitext(os.path.basename(image_path))
        processes = []  # List to store subprocess objects

        while i <= 360:
            orientation = f" {i} 0 0" 
            new_file_name = f"{file_name}_{i}_d{file_extension}"
            cmd = f"{renderer_path} {new_file_name} {parameters} {orientation} {width} {height}"
            i += rotate_degrees
            print("Running:", cmd)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            output = stdout.decode()
            error = stderr.decode()


            process.wait()
                        # Print or process the output and error
            print("Output:", output)
            print("Error:", error)
            # Decode the output and error (as they are in bytes format)
            

    except Exception as e:
        print("Error:", str(e))
        return -1
        
        
def render_result(renderer_path, image_path, angle, lat, long, alt, fov, pitch, roll):
    try:
        img = Image.open(image_path)
        width, height = img.size  
        width, height = scale_to_fit_screen(width, height, 1920, 1080)        
        parameters = f" {lat} {long} {alt} {fov}"
        i = 0
        file_name, file_extension = os.path.splitext(os.path.basename(image_path))
        processes = []  # List to store subprocess objects

        orientation = f" {angle} {pitch} {roll}"
        new_file_name = f"{file_name}_result_d{file_extension}"
        cmd = f"{renderer_path} {new_file_name} {parameters} {orientation} {width} {height}"
        print("Running:", cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        process.wait()        
    except Exception as e:
        print("Error:", str(e))
        return -1

import numpy as np
import math


def calculate_camera_matrix(horizontal_fov, width, height):
    """Calculate the camera matrix.

    Args:
    horizontal_fov (float): Horizontal field of view in degrees.
    width (int): Width of the image sensor or image in pixels.
    height (int): Height of the image sensor or image in pixels.

    Returns:
    numpy.ndarray: Camera matrix.
    """
    # Convert horizontal FOV to radians
    horizontal_fov_rad = math.radians(horizontal_fov)

    # Calculate focal lengths
    fx = width / (2.0 * math.tan(horizontal_fov_rad / 2.0))
    vertical_fov_rad = 2.0 * math.atan(math.tan(horizontal_fov_rad / 2.0) * height / width)
    fy = height / (2.0 * math.tan(vertical_fov_rad / 2.0))

    # Calculate optical center
    cx = width / 2.0
    cy = height / 2.0

    # Construct camera matrix
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])
    return camera_matrix
def rotation_matrix_to_pitch_yaw_roll(H):
    sy = np.sqrt(H[0,0] * H[0,0] +  H[1,0] * H[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(H[2,1] , H[2,2])
        y = np.arctan2(-H[2,0], sy)
        z = np.arctan2(H[1,0], H[0,0])
    else:
        x = np.arctan2(-H[1,2], H[1,1])
        y = np.arctan2(-H[2,0], sy)
        z = 0

    roll = np.degrees(x)
    pitch = np.degrees(y)
    yaw = np.degrees(z)
    print(f"pitch:{pitch} roll:{roll} yaw:{yaw}")
    return pitch, yaw, roll
    
def start_matching(image_path, rotate_degrees, fov):
    # Open the database.
    db = COLMAPDatabase.connect("testdatabase.db")
    db.create_tables()

    img = Image.open(image_path)
    width, height = img.size
    screenwidth, screenheight = scale_to_fit_screen(width, height, 1920, 1080)
    camera_matrix = calculate_camera_matrix(fov, screenwidth, screenheight)
    i = 0
    target_size = (int(screenwidth/2), int(screenheight/2))
    print(target_size)
    original_path = image_path
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    model11, width1, height1, params1 = (
        "SIMPLE_RADIAL",
        width,
        height,
        np.array((1024.0, 512.0, 384.0)),
    )
    model12, width2, height2, params2 = (
        "SIMPLE_RADIAL",
        screenwidth,
        screenheight,
        np.array((1024.0, 512.0, 384.0)),
    )
    camera_id1 = db.add_camera(model11, width1, height1, params1)
    camera_id2 = db.add_camera(model12, width2, height2, params2)
    image_id_original = db.add_image(f"{file_name}{file_extension}", camera_id1)
    
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load the original image in grayscale
    best_match_image_path = ""
    best_match_image_degree = 0.0
    best_match_prob = 0.0
    best_match_yaw = 0.0
    best_match_pitch = 0.0
    best_roll = 0.0
    best_match_image_deg = 0
    best_match_h = None
    device = K.utils.get_cuda_device_if_available()
    allkeypoints = np.empty((0,2))
    if not os.path.exists('./matches'):
        os.makedirs('matches')
    while i <= 360:
        new_image_path = f"rendered_images/{file_name}_{i}_d{file_extension}"
        matched_image_path = f"matches/{file_name}_{i}_d_matched{file_extension}"
        image_id = db.add_image(f"{file_name}_{i}_d{file_extension}", camera_id2)

        img1 = load_torch_image(image_path, target_size)
        img2 = load_torch_image(new_image_path, target_size)

        matcher = KF.LoFTR(pretrained='outdoor')

        input_dict = {"image0": K.color.rgb_to_grayscale(img1), # LofTR works on grayscale images only 
                      "image1": K.color.rgb_to_grayscale(img2)}

        with torch.no_grad():
            correspondences = matcher(input_dict)
        mkpts0 = correspondences['keypoints0'].numpy() * 2
        mkpts1 = correspondences['keypoints1'].numpy() * 2
        mkpts0[:, 0] *= height/screenheight
        mkpts0[:, 1] *= width/screenwidth
        if mkpts0.size < 10 or mkpts1.size < 10:
            i += rotate_degrees
            continue
        db.add_keypoints(image_id, mkpts1)
       
        H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        if H is None:
            i+=rotate_degrees
            continue

        matches = np.where(inliers.ravel() == 1)[0]
        print( mkpts0.size)
        matches =np.array([matches + allkeypoints.size/2,matches]).T
        print(allkeypoints.size/2)
        print(f"matches: {matches.size}")
        allkeypoints = np.vstack((allkeypoints, mkpts0))
        #E, mask = cv2.findEssentialMat(mkpts0, mkpts1, cameraMatrix)
        #retval, R, t, mask = cv2.recoverPose(E, mkpts0, mkpts1, cameraMatrix)
        #rotation_matrix_to_pitch_yaw_roll(R)
        db.add_matches(image_id_original, image_id, matches)
        inliers = inliers > 0
        if(inliers.size < 0):
            i += rotate_degrees
            continue
        if H is None or inliers is None:
            i += rotate_degrees
            continue
        match_prob = inliers.size
        #theta = np.arctan2(H[1, 0], H[0, 0])
        #yaw = np.degrees(theta)  # Convert radians to degrees
        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, camera_matrix)
        pitch = 0
        yaw = 0
        roll = 0
        for rotationmatrix in Rs:
            pitch, yaw, roll = rotation_matrix_to_pitch_yaw_roll(rotationmatrix)
        if match_prob > best_match_prob or best_match_image_path == "":
            best_match_image_path = new_image_path
            best_match_yaw = yaw
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
    db.add_keypoints(image_id_original, allkeypoints)
    db.commit()
    db.close()
    print(f"Best Match Image Path: {best_match_image_path}")
    print(f"Rotation Angle: {best_match_yaw} degrees")
    if best_match_h is not None:
        # Load the best match image
        best_match_image = cv2.imread(best_match_image_path)
        # Load the original image
        original_image_color = cv2.imread(original_path)
        # Assuming the original image is not grayscale because we're going to overlay it
        h, w = original_image_color.shape[:2]
        # Apply the warpPerspective function with the correct parameters
        warped_image = cv2.warpPerspective(best_match_image, best_match_h, (w, h), flags = cv2.WARP_INVERSE_MAP)

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
    return best_match_yaw + best_match_image_deg, best_match_pitch, best_roll

if __name__ == "__main__":
    os.chdir('../build-peakfinder-Desktop_Qt_6_7_0_MinGW_64_bit-Release/plain_renderer')
    # Path to plain_renderer.exe
    renderer_path = "plain_renderer.exe "
    image_path = select_image()
    if os.path.exists("testdatabase.db"):
        os.remove("testdatabase.db")
    if not image_path:
        print("No image selected. Exiting.")
        exit()

    info_path = change_extension_to_txt(image_path, ('photo_', 'info_'))
    if info_path is not None:
        lat, long, height, fov = read_info(info_path)
    
    if lat is not None:
        fov = math.degrees(fov)
    else:
        lat, long, height, fov = readExif(image_path)   
    rotate_degrees = round(fov-5)
    # Start the renderer with the specified parameters and select an image
    start_renderer(renderer_path,  image_path, rotate_degrees, lat, long, height, fov)
    
    yaw, pitch, roll = start_matching(image_path, rotate_degrees, fov)
    
    render_result(renderer_path, image_path, yaw, lat, long, height, fov, pitch, roll)
    
    
