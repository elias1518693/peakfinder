import tkinter as tk
import exifread
from tkinter import filedialog
import math
import numpy as np
import kornia.feature as KF
import kornia as K
from kornia_moons.feature import *
from PIL import Image
import matplotlib.pyplot as plt
import torch
from scipy.optimize import minimize
import os
import subprocess
import cv2

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
    img = cv2.imread(fname)
    img = cv2.resize(img, target_size)
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img

def decode_byte_array(byte_array, width, height, debug = False):
    # Assuming the format is RGBA32F, which has 4 channels (RGBA) of 32-bit floats
    num_channels = 4

    raw_data = bytes(byte_array)
    image_array = np.frombuffer(raw_data, dtype=np.float32)
    image_array = image_array.reshape((height, width, num_channels))
    # Flip the image vertically
    image_array = np.flipud(image_array)
    if(debug):
        #Output byte array as image for Debugging
        image = np.clip(image_array * 255, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        cv2.imshow("Decoded Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image_array
def select_image():
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an image file", parent=root)
    root.destroy()
    return file_path
    
def change_extension_to_txt(file_path, base_name_replace):
    directory, filename = os.path.split(file_path)
    name, _ = os.path.splitext(filename)
    new_name = name.replace(*base_name_replace) + '.txt'
    return os.path.join(directory, new_name)   

def yaw_to_north(yaw_radians):
    """
    Converts yaw from radians to degrees and rotates it by 90 degrees so that west becomes north
    """
    # Convert from radians to degrees
    yaw_degrees = yaw_radians * (180 / math.pi)

    yaw_degrees_adjusted = yaw_degrees + 90

    # Normalize to 0 - 360 degrees range
    yaw_degrees_normalized = yaw_degrees_adjusted % 360

    return yaw_degrees_normalized
   
def read_info(file_path):
    """
    Reads the intrinisc and extrinisc parameters from a file provided by geoPose3k database
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # Assuming the file structure is consistent and correct
            fov = float(lines[5].strip())
            height = float(lines[4].strip())
            original_latitude = float(lines[2].strip())
            original_longitude = float(lines[3].strip())
            print(yaw_to_north(float(lines[1].strip().split()[0])))
            return original_latitude, original_longitude, height, fov
    except Exception as e:
        print("Error:", str(e))
        return None, None, None, None


def scale_while_keeping_aspect(image_width, image_height, screen_width, screen_height):
    image_aspect_ratio = image_width / image_height
    screen_aspect_ratio = screen_width / screen_height
    if image_aspect_ratio > screen_aspect_ratio:
        scaled_width = screen_width
        scaled_height = scaled_width / image_aspect_ratio
    else:
        scaled_height = screen_height
        scaled_width = scaled_height * image_aspect_ratio
    scaled_width = min(screen_width, int(scaled_width))
    scaled_height = min(screen_height, int(scaled_height))
    # Adjust dimensions to be divisible by 8 because loftr works better
    scaled_width = scaled_width - scaled_width % 8
    scaled_height = scaled_height - scaled_height % 8
    return scaled_width, scaled_height

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

    if 'iphone' in model.lower() or 'samsung' in model.lower() or 'nokia' in model.lower() or 'pixel' in model.lower():
        sensor_width = 6.16  # Assume sensor size for a typical smartphone camera in mm
        sensor_height = 4.62 # Typical sensor height for smartphones in mm
        print('phone camera')
    else:
        sensor_width = 35.9  # Assume sensor size for a full-frame camera in mm
        sensor_height = 24   # Typical sensor height for full-frame cameras in mm
        print('normal camera')
    
    width, height = image_dimensions
    if height > width:
        fov = 2 * math.degrees(math.atan2(sensor_height / 2, focal_length))
    else:
        fov = 2 * math.degrees(math.atan2(sensor_width / 2, focal_length))

    return fov
    

def readExif(file_path):
    with open(file_path, 'rb') as f:
        exif_data = exifread.process_file(f)
        if not exif_data.get('GPS GPSLatitude'):
            f.close()
        else:
            latitude = exif_data.get('GPS GPSLatitude')
            longitude = exif_data.get('GPS GPSLongitude')
            height = exif_data.get('GPS GPSAltitude')

            latitude_dd = dms_to_dd(latitude)
            longitude_dd = dms_to_dd(longitude)
            height_dd = height_to_dd(height)

            img = Image.open(file_path)
            dims = img.size         
            fov = calculate_fov(exif_data, dims)
            print(fov)
            return latitude_dd, longitude_dd, height_dd, fov




def render_360_degrees(renderer_path, image_path, image_width, image_height, latitude, longitude, altitude, field_of_view):
    base_file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    rendered_images = []
    byte_data = start_plain_renderer(renderer_path, base_file_name, latitude, longitude, altitude, field_of_view, 0, 0, 0, width, height, 0)
    bytes_per_image = image_width * image_height * 4 * 4
    # Split the byte data into individual images
    for i in range(0, len(byte_data), bytes_per_image):
        image_data = byte_data[i:i + bytes_per_image]
        decoded_image = decode_byte_array(image_data, image_width, image_height)  # Assuming this function is defined
        rendered_images.append(decoded_image)
    return rendered_images



def calculate_reprojection_error(fov, mkpts0, ws_array1, width, height, dist_coeffs):
    camera_matrix = calculate_camera_matrix(fov, width, height)

    rvec = np.array([[0], [0], [0]], dtype=float)
    tvec = np.array([[0], [0], [0]], dtype=float)
    _, rvec, tvec, inliers = cv2.solvePnPRansac(ws_array1, mkpts0, camera_matrix, dist_coeffs, rvec, tvec,
                                                useExtrinsicGuess=True, iterationsCount=1000, reprojectionError=1.0,
                                                confidence=0.9999, flags=cv2.SOLVEPNP_ITERATIVE)

    if inliers is not None:
        inlier_keypoints = mkpts0[inliers].squeeze()
        inlier_3d_points = ws_array1[inliers].squeeze()
        projected_points, _ = cv2.projectPoints(inlier_3d_points, rvec, tvec, camera_matrix, dist_coeffs)
        error = np.mean(np.linalg.norm(inlier_keypoints - projected_points.squeeze(), axis=1) ** 2)
    else:
        error = np.inf
    return error
def optimize_fov(mkpts0, ws_array1, rotation_vector0, translation_vector0, width, height, dist_coeffs, initial_guess=90):
    result = minimize(calculate_reprojection_error, x0=initial_guess, args=(mkpts0, ws_array1, width, height, dist_coeffs), method='Nelder-Mead')
    optimized_fov = result.x[0]
    return optimized_fov


def start_plain_renderer(renderer_path, filename, lat, long, alt, fov, yaw, pitch, roll ,width, height, panorama):
    try:
        fx, fy, cx, cy = calculate_camera_parameters(fov, width, height)
        parameters = f"{lat} {long} {alt} {fov}"
        orientation = f"{yaw} {pitch} {roll}"
        translation = f"{fx} {fy} {cx} {cy}"
        cmd = f"{renderer_path} {filename} {parameters} {orientation} {translation} {width} {height} {panorama}"
        print("Running:", cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        output = stdout.decode()
        error = stderr.decode()
        byte_data = bytes.fromhex(output.strip())
        return byte_data
    except Exception as e:
        print("Error:", str(e))
        return -1

def render_result(renderer_path, image_path, width, height, yaw, lat, long, alt, fov, pitch, roll, optimize = True):
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))
    if (optimize):
        new_file_name = f"{file_name}_result"
    else:
        new_file_name = f"{file_name}_result_optimized"
    byte_data = start_plain_renderer(renderer_path, new_file_name, lat, long, alt, fov, yaw, pitch, roll, width, height, 1)
    depth_image = decode_byte_array(byte_data, width, height, False)
    device = torch.device('cuda')
    matcher = KF.LoFTR(pretrained='outdoor')
    matcher = matcher.to(device).eval()

    img1 = load_torch_image(image_path, [width,height])
    img2 = load_torch_image(f"rendered_images/{new_file_name}_0.jpg", [width,height])
    input_dict = {"image0": K.color.rgb_to_grayscale(img1).to(device),  # LofTR works on grayscale images only
                  "image1": K.color.rgb_to_grayscale(img2).to(device)}
    with torch.no_grad():
        correspondences = matcher(input_dict)
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    draw_matches(mkpts0,mkpts1, img1, img2, None, f'matches/{file_name}_refined.jpg')
    inlier_points1 = mkpts1
    index_y = np.round(inlier_points1[:, 1]).astype(int)
    index_x = np.round(inlier_points1[:, 0]).astype(int)

    valid_indices = np.where(depth_image[index_y, index_x][:, :3].any(axis=1))
    mkpts0 = mkpts0[valid_indices]
    mkpts1 = mkpts1[valid_indices]
    ws_array1 = depth_image[index_y[valid_indices], index_x[valid_indices]][:, :3].astype(np.float32)

    dist_coeffs = np.zeros((4, 1))
    if (ws_array1.shape[0] < 4):
        return
    camera_matrix = calculate_camera_matrix(fov, width, height)
    # SolvePnP returns the rotation and translation vectors
    success0, rotation_vector0, translation_vector0, pose_inliers0 = cv2.solvePnPRansac(ws_array1, mkpts0,
                                                                                        camera_matrix, distCoeffs=None,
                                                                                        flags=cv2.SOLVEPNP_ITERATIVE,
                                                                                        confidence=0.999999,
                                                                                        reprojectionError=1,
                                                                                        iterationsCount=20000)
    success1, rotation_vector1, translation_vector1, pose_inliers1 = cv2.solvePnPRansac(ws_array1, mkpts1,
                                                                                        camera_matrix, distCoeffs=None,
                                                                                        flags=cv2.SOLVEPNP_ITERATIVE,
                                                                                        confidence=0.999999,
                                                                                        reprojectionError=1,
                                                                                        iterationsCount=20000)
    R1, _ = cv2.Rodrigues(rotation_vector0)
    R2, _ = cv2.Rodrigues(rotation_vector1)
    print(
        f'success: {success0} \n rotation vector: {rotation_vector0} \n  translation vector: {translation_vector0} \n')
    rotation_matrix_to_pitch_yaw_roll(R1)
    print(
        f'success: {success1} \n rotation vector: {rotation_vector1} \n  translation vector: {translation_vector1} \n')
    rotation_matrix_to_pitch_yaw_roll(R2)
    adjusted_translation1 = -np.matrix(R1).T * np.matrix(translation_vector0)
    adjusted_translation2 = -np.matrix(R2).T * np.matrix(translation_vector1)
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
    rotation_matrix_to_pitch_yaw_roll(relative_rotation)
    print(f'Relative rotation: {relative_rotation} \n relative translation: {relative_translation}')
    ws_array1 = ws_array1[pose_inliers0]
    mkpts0 = mkpts0[pose_inliers0]
    rotation_vector0, translation_vector0 = cv2.solvePnPRefineLM(ws_array1, mkpts0, camera_matrix, None, rotation_vector0, translation_vector0)
    print(
        f'success: {success0} \n rotation vector: {rotation_vector0} \n  translation vector: {translation_vector0} \n')
    R1, _ = cv2.Rodrigues(rotation_vector0)
    rotation_matrix_to_pitch_yaw_roll(R1)
    if(optimize):
        optimized_fov = optimize_fov(mkpts0, ws_array1, rotation_vector0, translation_vector0, width, height, dist_coeffs, fov)
        print(f'Optimized FoV: {optimized_fov}')
        render_result(renderer_path, image_path, width, height, yaw, lat, long, alt, optimized_fov, pitch, roll, False)

def calculate_camera_matrix(horizontal_fov_deg, width, height):
    horizontal_fov_rad = math.radians(horizontal_fov_deg)
    vertical_fov_rad = 2.0 * math.atan(math.tan(horizontal_fov_rad / 2.0) * height / width)
    fx = width / (2.0 * math.tan(horizontal_fov_rad / 2.0))
    fy = height / (2.0 * math.tan(vertical_fov_rad / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])
    return camera_matrix

def calculate_camera_parameters(horizontal_fov_deg, width, height):
    horizontal_fov_rad = math.radians(horizontal_fov_deg)
    vertical_fov_rad = 2.0 * math.atan(math.tan(horizontal_fov_rad / 2.0) * height / width)
    fx = width / (2.0 * math.tan(horizontal_fov_rad / 2.0))
    fy = height / (2.0 * math.tan(vertical_fov_rad / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy

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

def calculate_relative_transformation(R1, R2, adjusted_translation1, adjusted_translation2 ):
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
    print(f'Relative rotation: {relative_rotation} \n relative translation: {relative_translation}')
    return yaw, pitch, roll
def warp_image(file_name,scaled_image, rendered_image, H, width, height):

    # Apply the warpPerspective function with the correct parameters
    warped_image = cv2.warpPerspective(rendered_image, H, (width, height), flags=cv2.WARP_INVERSE_MAP)

    # Overlay the warped image onto the original image
    # You can adjust the alpha value to make the overlay transparent
    alpha = 0.5
    overlay_image = cv2.addWeighted(scaled_image, 1 - alpha, warped_image, alpha, 0)

    overlay_image_path = f"overlay_{file_name}.png"
    cv2.imwrite(overlay_image_path, overlay_image)

def start_matching(image_path, screenwidth, screenheight, fov, byte_array):
    fov_vert = math.degrees(2.0 * math.atan(math.tan(math.radians(fov) / 2.0) * screenheight / screenwidth))
    camera_matrix = calculate_camera_matrix(fov, screenwidth, screenheight)
    target_size = (int(screenwidth), int(screenheight))
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
    best_x, best_y, best_z = 0,0,0

    device = torch.device('cuda')
    matcher = KF.LoFTR(pretrained='outdoor')
    matcher = matcher.to(device).eval()
    input_image = load_torch_image(scaled_image_path, target_size)
    if not os.path.exists('./matches'):
        os.makedirs('matches')
    for i in range(int(360/fov_vert)):
        print(f'matching image {i} with {fov_vert * i} degrees:')
        new_image_path = f"rendered_images/{file_name}_{i}.jpg"
        matched_image_path = f"matches/{file_name}_{i}_matched.jpg"
        rendered_image = load_torch_image(new_image_path, target_size)
        input_dict = {"image0": K.color.rgb_to_grayscale(input_image).to(device), # LofTR works on grayscale images only
                      "image1": K.color.rgb_to_grayscale(rendered_image).to(device)}
        with torch.no_grad():
            correspondences = matcher(input_dict)
        mkpts_real = correspondences['keypoints0'].cpu().numpy()
        mkpts_rendered = correspondences['keypoints1'].cpu().numpy()

        if mkpts_real.shape[0] < 10 or mkpts_rendered.shape[0] < 10:
            continue
        print(f"matches: {mkpts_rendered.shape[0]}")
        index_y = np.round(mkpts_rendered[:, 1]).astype(int)
        index_x = np.round(mkpts_rendered[:, 0]).astype(int)

        valid_indices = np.where(byte_array[i][index_y, index_x][:, :3].any(axis=1))
        mkpts_real = mkpts_real[valid_indices]
        mkpts_rendered = mkpts_rendered[valid_indices]
        ws_array1 = byte_array[i][index_y[valid_indices], index_x[valid_indices]][:, :3].astype(np.float32)

        if(ws_array1.shape[0] < 4):
            continue

        # SolvePnP returns the rotation and translation vectors
        success0, rotation_vector_real, translation_vector_real, pose_inliers_real = cv2.solvePnPRansac(ws_array1, mkpts_real, camera_matrix, distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.9999, reprojectionError=1)
        success1, rotation_vector_rendered, translation_vector_rendered, pose_inliers_rendered = cv2.solvePnPRansac(ws_array1,  mkpts_rendered, camera_matrix, distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.9999, reprojectionError=1)

        if(not(success0 and success1)):
            continue
        #draw_matches(mkpts_real, mkpts_rendered, input_image, rendered_image, None, matched_image_path)

        R_real, _ = cv2.Rodrigues(rotation_vector_real)
        R_rendered, _ = cv2.Rodrigues(rotation_vector_rendered)

        print(
            f'success real image: {success0} \n rotation vector: {rotation_vector_real} \n  translation vector: {translation_vector_real} \n translation vector with R: {translation_vector_real}')
        print(
            f'success rendered image: {success1} \n rotation vector: {rotation_vector_rendered} \n translation vector: {translation_vector_rendered} \n translation vector with R: {translation_vector_rendered} This one should be 0')
        rotation_matrix_to_pitch_yaw_roll(R_real)
        rotation_matrix_to_pitch_yaw_roll(R_rendered)

        yaw, pitch, roll = calculate_relative_transformation(R_real, R_rendered, translation_vector_real, translation_vector_rendered)

        match_prob = calculate_reprojection_error(fov, mkpts_real, ws_array1, width, height, None)
        print(f"Reprojection error: {match_prob}")
        if match_prob > best_match_prob or best_match_image_path == "":
            best_match_image_path = new_image_path
            best_match_yaw = pitch
            best_match_pitch = -roll
            best_roll = yaw
            best_match_image_deg = i * fov_vert
            best_match_prob = match_prob
            best_match_i = i
            #best_x = -translation_vector0[1][0]
            #best_y = -translation_vector0[0][0]
            #best_z = -translation_vector0[2][0]


    print(f"Best Match Image Path: {best_match_image_path}")
    print(f"Rotation Angle: {best_match_yaw} degrees")
    return best_match_yaw + best_match_image_deg, best_match_pitch, best_roll, best_x, best_y, best_z, byte_array[best_match_i]

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

    img = Image.open(image_path)
    width, height = img.size
    width, height = scale_while_keeping_aspect(width, height, 960, 540)
    # Start the renderer with the specified parameters and select an image
    print(f'calculated fov: {fov}')

    byte_array = render_360_degrees(renderer_path,  image_path, width, height, lat, long, height, fov)

    yaw, pitch, roll, x, y, z, best_array = start_matching(image_path, width, height, fov, byte_array)
    
    render_result(renderer_path, image_path,width, height, yaw, lat, long, height, fov, pitch, roll)
    
    
