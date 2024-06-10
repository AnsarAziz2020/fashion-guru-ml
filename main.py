import json
import os

from collections import defaultdict
import PIL.Image
from flask import jsonify
from ultralytics import YOLO
import math
import cv2
from colorthief import ColorThief

current_dir = os.getcwd()
current_working_dir = os.path.abspath(os.path.join(current_dir))

IMAGES_DIR = os.path.join(current_working_dir, 'images')

shirt_tshirt_model_path = os.path.join('runs/tshirt_shirt/weights/best_color.pt')
shirt_tshirt_model = YOLO(shirt_tshirt_model_path)
type_shirt_model_path = os.path.join(current_working_dir, 'runs\\type_shirt\\weights\\last_multiple_2.pt')
type_shirt_model = YOLO(type_shirt_model_path)
threshold = 0.5

def check_Shirt_TShirt(image):
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_frame_gray = resizeImage(frame_gray)
    # Replicate the grayscale image into three channels to simulate an RGB image
    frame = cv2.cvtColor(resized_frame_gray, cv2.COLOR_GRAY2BGR)
    H, W = resized_frame_gray.shape
    results = shirt_tshirt_model(frame)[0]
    print(results.boxes.data.tolist())
    return (results.boxes.data.tolist())

def check_Type_Shirt(image):
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_frame_gray = resizeImage(frame_gray)
    # Replicate the grayscale image into three channels to simulate an RGB image
    frame = cv2.cvtColor(resized_frame_gray, cv2.COLOR_GRAY2BGR)
    H, W = resized_frame_gray.shape
    results = type_shirt_model(frame)[0]
    print(results.boxes.data.tolist())
    return results.boxes.data.tolist()


def resizeImage(image):
    new_width = 300
    original_height, original_width = image.shape
    scale_factor = new_width / original_width
    new_height = int(original_height * scale_factor)
    return cv2.resize(image, (new_width, new_height))

def resizeImagePIL(original_width,original_height):
    new_width = 300
    scale_factor = new_width / original_width
    new_height = int(original_height * scale_factor)
    return (new_width,new_height)

def getDominantColor(box,bytes_io):
    image = PIL.Image.open(bytes_io)
    image = image.resize((resizeImagePIL(image.width,image.height)))
    x1, y1, x2, y2 = box
    image = image.crop(box)
    width, height = image.size
    lower_height=(height/2)-(height*0.10)
    higher_height = (height / 2) + (height * 0.10)
    colors_frequency = {}
    # Line point getting two lines and extract their color
    for x in range(0,width-2,2):
        color_upper_left = image.getpixel((x, lower_height))
        color_upper_right = image.getpixel((x, lower_height))
        color_lower_left = image.getpixel((x, higher_height))
        color_lower_right = image.getpixel((x, higher_height))
        combined_color = tuple((c1 + c2 + c3 +c4) // 4 for c1, c2 , c3, c4 in zip(color_upper_left,color_upper_right,color_lower_left,color_lower_right))
        if combined_color in colors_frequency:
            colors_frequency[combined_color[0:3]] += 1
        else:
            colors_frequency[combined_color[0:3]] = 1

    merged_colors_frequency = merge_nearest_colors(colors_frequency, 25)
    merged_colors_sorted = sorted(merged_colors_frequency.items(), key=lambda x: x[1], reverse=True)

    final_merged_color_list = list(map(lambda x:x[0],merged_colors_sorted))

    bg_color_list=[]
    bg_color_list.append(image.getpixel((0, higher_height))[0:3])
    bg_color_list.append(image.getpixel((0, lower_height))[0:3])
    bg_color_list.append(image.getpixel((width-1, lower_height))[0:3])
    bg_color_list.append(image.getpixel((width - 1, higher_height))[0:3])
    color_bg_threshold = 10

    for color in final_merged_color_list:
        R,G,B= color
        for bg_color in bg_color_list:
            R_bg, G_bg , B_bg = bg_color
            if(R+color_bg_threshold>R_bg or R-color_bg_threshold>R_bg):
                if (G + color_bg_threshold > G_bg or G - color_bg_threshold > G_bg):
                    if (B + color_bg_threshold > B_bg or B - color_bg_threshold > B_bg):
                        if color in final_merged_color_list:
                            final_merged_color_list.remove(color)
    return json.dumps(final_merged_color_list[0:2])


def euclidean_distance(color1, color2):
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))


def merge_nearest_colors(colors_frequency, threshold):
    merged_colors_frequency = defaultdict(int)

    for color1, count1 in colors_frequency.items():
        merged = False
        for color2, count2 in merged_colors_frequency.items():
            if euclidean_distance(color1, color2) <= threshold:
                merged_colors_frequency[color2] += count1
                merged = True
                break
        if not merged:
            merged_colors_frequency[color1] += count1

    return merged_colors_frequency

