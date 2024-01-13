import subprocess
import tkinter as tk
import exifread
import os
import socket
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.optim as optim



def project_points_opencv(points_3d, camera_params, image_size):


    # Assuming no lens distortion
    dist_coeffs = np.zeros(4)

    # OpenCV's projectPoints expects rotation and translation vectors
    rvec = np.zeros(3)
    tvec = np.zeros(3)

    # Convert points_3d to the correct shape and type if necessary
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 1, 3)

    # Project points
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_params, dist_coeffs)

    # Reshape the output and return
    return points_2d.reshape(-1, 2)

def project_points(points_3d_tensor, camera_params_tensor, image_size):
    """
       Wrapper function to interface with PyTorch.
       """
    # Convert PyTorch tensors to numpy arrays
    points_3d_np = points_3d_tensor.detach().cpu().numpy()
    camera_params_np = camera_params_tensor.detach().cpu().numpy()

    # Use OpenCV function for projection
    points_2d_np = project_points_opencv(points_3d_np, camera_params_np, image_size)

    # Convert back to PyTorch tensor
    points_2d_tensor = torch.from_numpy(points_2d_np).type_as(points_3d_tensor)

    return points_2d_tensor



def plot_3d_points(points, special_point1, special_point2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extracting x, y, and z coordinates from points
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    z_coords = [point[2] for point in points]

    # Plotting regular points
    ax.scatter(x_coords, y_coords, z_coords)

    # Plotting special points in red
    ax.scatter(*special_point1, color='green', s=100)
    ax.scatter(*special_point2, color='red', s=100)

    # Labeling axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()


def plot_3d_color(points, img):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=0, azim=-90)
    # Extracting x, y, and z coordinates from points
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    z_coords = [point[2] for point in points]
    color = img
    # Plotting regular points
    ax.scatter(x_coords, y_coords, z_coords, color = color, s =1)

    # Plotting special points in red
    ax.scatter(*[0,0,0], color = 'red', s=100)

    # Labeling axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()
def load_torch_image(fname, target_size):
    # Load the image using OpenCV
    img = cv2.imread(fname)
    
    # Resize the image using OpenCV
    img = cv2.resize(img, target_size)
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img
    
    return img

def decode_byte_array(byte_array, width, height):
    # Assuming the format is RGBA32F, which has 4 channels (RGBA) of 32-bit floats
    num_channels = 4

    # Convert QByteArray to bytes (this step may vary depending on how you receive the data)
    raw_data = bytes(byte_array)

    # Create a NumPy array from the raw data
    image_array = np.frombuffer(raw_data, dtype=np.float32)

    # Reshape the array to match the image dimensions and channels
    # The order is height, width, and then number of channels
    image_array = image_array.reshape((height, width, num_channels))

    # Flip the image vertically
    image_array = np.flipud(image_array)

    #image = np.clip(image_array * 255, 0, 255).astype(np.uint8)

    # Convert from RGBA to BGR format for OpenCV
    #image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    #cv2.imshow("Decoded Image", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return image_array
def select_image():
    root = tk.Tk()
    root.attributes("-topmost", True)  # This makes the window appear on top
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(title="Select an image file", parent=root)

    # Destroy the root window after selection
    root.destroy()
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

    # Adjust dimensions to be divisible by 8
    scaled_width = scaled_width - scaled_width % 8
    scaled_height = scaled_height - scaled_height % 8

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

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return dim
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


def start_renderer(renderer_path, image_path, lat, long, alt, fov):
    try: 
        img = Image.open(image_path)
        width, height = img.size
        fov_vert = math.degrees(2.0 * math.atan(math.tan(math.radians(fov) / 2.0) * height / width))
        print(f'horizontal fov: {fov} vertical fov: {fov_vert}')
        width, height = scale_to_fit_screen(width, height, 960, 540)
        file_name, file_extension = os.path.splitext(os.path.basename(image_path))
        orientation = f"0 0 0"
        translation = f"0 0 0"
        cmd = f"{renderer_path} {file_name} {lat} {long} {alt} {fov} {orientation} {translation} {width} {height} {0}"
        print("Running:", cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        output = stdout.decode()
        error = stderr.decode()

        position_images = []
        # Convert the line to a bytes object
        byte_data = bytes.fromhex(output.strip())
        bytes_per_image = width * height * 4 * 4  # Modify this if your images are not 1 byte per pixel

        # Split byte_data into individual images
        split_byte_data = [byte_data[i:i + bytes_per_image] for i in range(0, len(byte_data), bytes_per_image)]

        for i, byte_array_str in enumerate(split_byte_data):
            # Assuming decode_byte_array is a defined function
            image = decode_byte_array(byte_array_str, width, height)
            color_img = cv2.imread(f"rendered_images/{file_name}_{i}.jpg")
            color_img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            colors = color_img_rgb.reshape(-1, 3) / 255.0
            #plot_3d_color(image.reshape((height * width, 4)[:3]), colors)
            position_images.append(image)

        return position_images


    except Exception as e:
        print("Error:", str(e))
        exit()
        
        
def render_result(renderer_path, image_path, yaw, lat, long, alt, fov, pitch, roll, x, y, z):
    try:
        img = Image.open(image_path)
        width, height = img.size  
        width, height = scale_to_fit_screen(width, height, 960, 540)
        parameters = f" {lat} {long} {alt} {fov}"
        file_name, file_extension = os.path.splitext(os.path.basename(image_path))
        orientation = f"{yaw} {pitch} {roll}"
        translation = f"{x} {y} {z}"
        new_file_name = f"{file_name}_result"
        cmd = f"{renderer_path} {new_file_name} {parameters} {orientation} {translation} {width} {height} {1}"
        print("Running:", cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    except Exception as e:
        print("Error:", str(e))
        return -1




def calculate_camera_matrix(horizontal_fov, width, height):
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
    #print(f'camera matrix: {camera_matrix}')
    return camera_matrix

def draw_matches(mkpts0, mkpts1, img1, img2, inliers, path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1, -1, 2),
                                     torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                                     torch.ones(mkpts0.shape[0]).view(1, -1, 1)),
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1, -1, 2),
                                     torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                                     torch.ones(mkpts1.shape[0]).view(1, -1, 1)),
        torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
        K.tensor_to_image(img1),
        K.tensor_to_image(img2),
        inliers,
        draw_dict={'inlier_color': None,
                   'tentative_color': None,
                   'feature_color': (0.2, 0.5, 1), 'vertical': False},
        ax=ax, )
    plt.savefig(path)
def rotation_matrix_to_pitch_yaw_roll(H):
    sy = np.sqrt(H[0,0] * H[0,0] +  H[1,0] * H[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(-H[2,1] , H[2,2])
        y = np.arcsin(H[2,0])
        z = np.arctan2(-H[1,0], H[0,0])
    else:
        x = np.arctan2(-H[1,2], H[1,1])
        y = np.arctan2(-H[2,0], sy)
        z = 0

    roll = np.degrees(x)
    pitch = np.degrees(y)
    yaw = np.degrees(z)
    print(f"yaw:{yaw} pitch:{pitch} roll:{roll} ")
    return yaw, pitch, roll

def rot_params_rv(rvecs):
    R = cv2.Rodrigues(rvecs)[0]
    roll = 180*math.atan2(-R[2][1], R[2][2])/math.pi
    pitch = 180*math.asin(R[2][0])/math.pi
    yaw = 180*math.atan2(-R[1][0], R[0][0])/math.pi
    print(f"yaw:{yaw} pitch:{pitch} roll:{roll} ")
    return pitch, yaw, roll
def start_matching(image_path, fov, byte_array):
    img = Image.open(image_path)
    width, height = img.size
    screenwidth, screenheight = scale_to_fit_screen(width, height, 960, 540)
    fov_vert = math.degrees(2.0 * math.atan(math.tan(math.radians(fov) / 2.0) * screenheight / screenwidth))
    camera_matrix = calculate_camera_matrix(fov, screenwidth, screenheight)
    target_size = (int(screenwidth), int(screenheight))
    print(target_size)
    original_path = image_path
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    scaled_image_path = f"rendered_images/{file_name}_scaled.jpg"
    scaled_image = cv2.resize(cv2.imread(image_path), (int(screenwidth), int(screenheight)))
    cv2.imwrite(scaled_image_path, scaled_image)

    best_match_image_path = ""
    best_match_prob = 0.0
    best_match_yaw = 0.0
    best_match_pitch = 0.0
    best_roll = 0.0
    best_match_image_deg = 0
    best_match_h = None
    best_x, best_y, best_z = 0,0,0
    allkeypoints = np.empty((0,2))
    device = torch.device('cuda')
    matcher = KF.LoFTR(pretrained='outdoor')
    matcher = matcher.to(device).eval()
    if not os.path.exists('./matches'):
        os.makedirs('matches')
    for i in range(int(360/fov_vert)):
        print(f'matching image {i} with {fov_vert * i} degrees:')
        new_image_path = f"rendered_images/{file_name}_{i}.jpg"
        matched_image_path = f"matches/{file_name}_{i}_matched.jpg"

        img1 = load_torch_image(image_path, target_size)
        img2 = load_torch_image(new_image_path, target_size)

        input_dict = {"image0": K.color.rgb_to_grayscale(img1).to(device), # LofTR works on grayscale images only
                      "image1": K.color.rgb_to_grayscale(img2).to(device)}

        with torch.no_grad():
            correspondences = matcher(input_dict)
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        if mkpts0.size < 10 or mkpts1.size < 10:
            continue

        H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 1, 0.999999, 500000)
        if H is None:
            continue

        matches = np.where(inliers.ravel() == (1))[0]
        matches =np.array([matches + allkeypoints.size/2,matches]).T
        print(f"matches: {matches.size}")
        inlier_points1 = mkpts1
        ws_array1 = np.zeros((inlier_points1.shape[0], 3), dtype=np.float32)
        for idx, keypoint in enumerate(inlier_points1):
            # Convert keypoint coordinates to integers
            x_index = round(keypoint[0])
            y_index = round(keypoint[1])

            if(byte_array[i][y_index, x_index][:3].any()):
                #print(byte_array[i][y_index, x_index])
                ws_array1[idx] = byte_array[i][y_index, x_index][:3]
            else:
                print(F'{byte_array[i][y_index, x_index]}  x: {y_index} y: {x_index}')


        dist_coeffs = np.zeros((4, 1))
        if(ws_array1.shape[0] < 4):
            continue

        # SolvePnP returns the rotation and translation vectors
        success0, rotation_vector0, translation_vector0, pose_inliers0 = cv2.solvePnPRansac(ws_array1, mkpts0, camera_matrix, distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.9999, reprojectionError=1)
        success1, rotation_vector1, translation_vector1, pose_inliers1 = cv2.solvePnPRansac(ws_array1,  mkpts1, camera_matrix, distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.9999, reprojectionError=1)
        E, mask = cv2.findEssentialMat(mkpts0, mkpts1)

        #rotation_vector1 = np.array([[0.0],[0.0],[0.0]])
        #translation_vector1 = np.array([[0.0],[0.0],[0.0]])


        if(not(success0 and success1)):
            continue
        draw_matches(mkpts0, mkpts1, img1, img2, inliers, matched_image_path)
        #plot_3d_points(ws_array1, -translation_vector0, -translation_vector1)
        all_points = byte_array[i][:,:,:3]
       # plot_3d_points(all_points, -translation_vector0, -translation_vector1)
        R1, _ = cv2.Rodrigues(rotation_vector0)
        R2, _ = cv2.Rodrigues(rotation_vector1)
        adjusted_translation1 = -np.matrix(R1).T * np.matrix(translation_vector0)
        adjusted_translation2 = -np.matrix(R2).T * np.matrix(translation_vector1)
        print(
            f'success: {success0} \n rotation vector: {rotation_vector0} \n  translation vector: {translation_vector0} \n translation vector with R: {adjusted_translation1}')
        print(
            f'success: {success1} \n rotation vector: {rotation_vector1} \n translation vector: {translation_vector1} \n translation vector with R: {adjusted_translation2} This one should be 0')
        rotation_matrix_to_pitch_yaw_roll(R1)
        rotation_matrix_to_pitch_yaw_roll(R2)
        T1 = np.hstack((R1, adjusted_translation1))
        T2 = np.hstack((R2, adjusted_translation2))

        # Convert to 4x4 transformation matrices
        T1 = np.vstack((T1, [0, 0, 0, 1]))
        T2 = np.vstack((T2, [0, 0, 0, 1]))

        # Compute relative transformation
        M = np.dot(T2, np.linalg.inv(T1))

        # Extract relative rotation (R) and translation (T)
        relative_rotation = M[:3, :3]
        relative_translation = M[:3, 3]
        yaw, pitch, roll = rotation_matrix_to_pitch_yaw_roll(relative_rotation)
        print( f'Relative rotation: {relative_rotation} \n relative translation: {relative_translation}')

        objectPoints = np.array([ws_array1], dtype=np.float32)

        # Ensure imagePoints0 and imagePoints1 are lists of arrays, one array for each image pair
        # Convert them to np.float32 if not already
        imagePoints0 = np.array([np.float32(mkpts0) for _ in range(len(objectPoints))], dtype=np.float32)
        imagePoints1 = np.array([np.float32(mkpts1) for _ in range(len(objectPoints))], dtype=np.float32)

        # Perform stereo calibration
       # flags = (cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_USE_INTRINSIC_GUESS)
      #  ret, K1, D1, K2, D2, R, T, Ess, F = cv2.stereoCalibrate(
       #     objectPoints, imagePoints0, imagePoints1,
        #    camera_matrix, dist_coeffs, camera_matrix, dist_coeffs,
       #     [screenwidth, screenheight], flags)
      #  print(f"Stereo calibration rms: {ret}")
        if(matches.size < 0):
            continue
        match_prob = ws_array1.size
        # theta = np.arctan2(H[1, 0], H[0, 0])
        # yaw = np.degrees(theta)  # Convert radians to degrees
        num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, camera_matrix)
        # for rotationmatrix in Rs:
        # yaw, pitch, roll = rotation_matrix_to_pitch_yaw_roll(rotationmatrix)
        if match_prob > best_match_prob or best_match_image_path == "":
            best_match_image_path = new_image_path
            best_match_yaw = pitch
            best_match_pitch = -roll
            best_roll = yaw
            best_match_image_deg = i * fov_vert
            best_match_prob = match_prob
            best_match_h = H
            best_points1 = mkpts0
            best_points2 = mkpts1
            best_camera_matrix =camera_matrix
            best_pos_array = ws_array1
            best_rotation = R2
            best_translation = translation_vector1
            #best_x = -translation_vector0[1][0]
            #best_y = -translation_vector0[0][0]
            #best_z = -translation_vector0[2][0]


    print(f"Best Match Image Path: {best_match_image_path}")
    print(f"Rotation Angle: {best_match_yaw} degrees")
    if best_match_h is not None:
        # Load the best match image
        best_match_image = cv2.imread(best_match_image_path)

        # Assuming the original image is not grayscale because we're going to overlay it
        h, w = scaled_image.shape[:2]
        # Apply the warpPerspective function with the correct parameters
        warped_image = cv2.warpPerspective(best_match_image, best_match_h, (w, h), flags = cv2.WARP_INVERSE_MAP)

        # Overlay the warped image onto the original image
        # You can adjust the alpha value to make the overlay transparent
        alpha = 0.5
        overlay_image = cv2.addWeighted(scaled_image, 1 - alpha, warped_image, alpha, 0)

        # Save or show the overlay image
        overlay_image_path = f"overlay_{file_name}.png"
        cv2.imwrite(overlay_image_path, overlay_image)  # Saving the overlay image
        # cv2.imshow("Overlay Image", overlay_image)  # If you want to display the image
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()  # Make sure to destroy all windows if you've used cv2.imshow

    min_error = cv2.projectPoints(best_pos_array, best_rotation, best_translation, camera_matrix, dist_coeffs)
    for guessfov in np.arange(30, 120, 0.1):
        # Project 3D points to 2D image points using the test_camera_matrix
        test_camera_matrix = calculate_camera_matrix(guessfov, screenwidth, screenheight)
        projected_points, _ = cv2.projectPoints(best_pos_array, best_rotation, best_translation, test_camera_matrix, dist_coeffs)

        # Calculate reprojection error
        reprojection_error = np.linalg.norm(best_points1 - projected_points.squeeze(), axis=1).mean()
        print(f'error for {guessfov} is {reprojection_error}')
        if(reprojection_error < min_error):
            min_error = reprojection_error
            best_fov = guessfov
    print(best_fov)

    return best_match_yaw + best_match_image_deg, best_match_pitch, best_roll, best_x, best_y, best_z

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

    # Start the renderer with the specified parameters and select an image
    byte_array = start_renderer(renderer_path,  image_path, lat, long, height, fov)

    yaw, pitch, roll, x, y, z = start_matching(image_path, fov, byte_array)
    
    render_result(renderer_path, image_path, yaw, lat, long, height, fov, pitch, roll, x, y, z)
    
    
