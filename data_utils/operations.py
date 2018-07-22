import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import cv2
import os
import random
from math import ceil, floor, pi, sqrt

from data_utils.write_xml_file import write_xml_file
from data_utils.constants import LABEL, IMAGE_SIZE, CARD_HEIGHT, CARD_WIDTH

img_placeholder = None
resize_placeholder = None
tf_img = None
sess = None

def instantiate_global_variables():
    global img_placeholder, resize_placeholder, tf_img, sess

    tf.reset_default_graph()
    img_placeholder = tf.placeholder(tf.float32, (None, CARD_HEIGHT, CARD_WIDTH, 3))
    resize_placeholder = tf.placeholder(tf.int32, (2))   # Resized height and width
    tf_img = tf.image.resize_images(img_placeholder, resize_placeholder,
                                        tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())



def tf_resize_images_with_white_bg(img, image_width, image_height, white_area_percent):
    # Estimate coordinates of the image inside white border based percentage area of white border.
    box_area = image_width * image_height
    resized_area = (100 - white_area_percent) * box_area / 100.0
    resized_width = ceil(image_width * sqrt(resized_area / box_area))
    resized_height = ceil(image_height * sqrt(resized_area / box_area))
    width_start = ceil((image_width - resized_width) / 2.0)
    height_start = ceil((image_height - resized_height) / 2.0)

    resized_img = sess.run(tf_img, feed_dict = {img_placeholder: img, 
                                                resize_placeholder: [resized_height, resized_width]})
       
    return_img = np.ones((len(resized_img), image_height, image_width, 3), dtype = np.float32)
    for index, r_img in enumerate(resized_img):
        return_img[index, height_start : height_start + resized_height,
                   width_start : width_start + resized_width, :] = r_img[:, :, :]
    return return_img

# Rotation is not being used currently. Use this method if it shows improvement in results.
def tf_rotate_images(img, angle_at):
    radian = angle_at * pi / 180
    tf.reset_default_graph()
    tf_rotate_img = tf.contrib.image.rotate(img, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rotate_img = sess.run(tf_rotate_img)
    return rotate_img

def save_img_as_png(X_file_data, file_name, folder_name):
    file_name_comp = file_name.split('.')
    new_file_name = '{}/{}.png'.format(folder_name, '_'.join(file_name_comp[:-1]))
    mpimg.imsave(new_file_name, X_file_data)
    
def fetch_image_files(label_list):
    image_array = []
    for file_path in label_list:
        img = mpimg.imread(file_path)[:, :, :3]  # Don't include alpha channel.
        image_array.append(img)
    image_array = np.array(image_array)
    return image_array

def add_salt_pepper_noise(image):
    row,col,ch = image.shape
    s_vs_p = 0.25
    amount = 0.004
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    image[coords[0], coords[1], :] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    image[coords[0], coords[1], :] = 0
    return image

def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.random((row, col, 1)).astype(np.float32)
    gauss = np.concatenate((gauss, gauss, gauss), axis = 2)
    noisy = cv2.addWeighted(image, 0.8, 0.2 * gauss, 0.2, 0)
    return noisy

def tf_generate_images(card_file_names, card_labels, bg_img_folder, save_img_folder, save_xml_folder, parameter_dict):
    # Extract the parameter values from the dictionary.
    scale_at = parameter_dict['scale_at']
    angle_at = parameter_dict['angle_at']
    background_index_at = parameter_dict['background_index_at']
    save_index = parameter_dict['save_index']
    raw_card_size = parameter_dict['raw_card_size']
    no_of_files_array = parameter_dict['no_of_files_array']
    border_area = parameter_dict['border_area']
    bg_total_files = parameter_dict['bg_total_files']

    # Fetch the card images and resize them.
    card_files = fetch_image_files(card_file_names)
    card_height, card_width = raw_card_size
    card_size = (ceil(scale_at * card_height), ceil(scale_at * card_width))
    card_files_op = tf_resize_images_with_white_bg(card_files, card_size[1], card_size[0], border_area)
    # Angle rotation is not giving any benefits. Hence disable it at the moment.
    #card_files_op = tf_rotate_images(card_files_op, angle_at)
    card_file_shape = card_files_op[0].shape

    # Various index counters
    file_index_at = 0
    current_background_index = background_index_at
    current_save_index = save_index
    card_index_at = 0

    cwd = os.getcwd()
    for i in range(len(no_of_files_array)):
        current_background_index += 1
        if current_background_index >= bg_total_files:
            current_background_index = 1
        background_file = '{}/{:06d}.png'.format(bg_img_folder, current_background_index)
        background_img = mpimg.imread(background_file)[:, :, :3]

        
        no_of_files = no_of_files_array[i]
        is_difficult_array, is_truncated_array, is_occluded_array = [], [], []
        boxes, image_card_labels = [], []
        for file_index in range(no_of_files):
            # Currently this supports only two images
            if no_of_files == 1:
                end_index_x, end_index_y = IMAGE_SIZE, IMAGE_SIZE
                start_index_x, start_index_y = 0, 0
            elif file_index == 0:
                end_index_x, end_index_y = floor(IMAGE_SIZE / 2.0), IMAGE_SIZE
                start_index_x, start_index_y = 0, 0
            elif file_index == 1:
                end_index_x, end_index_y = IMAGE_SIZE, IMAGE_SIZE
                start_index_x, start_index_y = ceil(IMAGE_SIZE / 2.0), 0
                                
            # Truncation
            should_truncate_choice = random.choice([True, False])
            # Avoiding Truncation in this case always
            should_truncate_choice = False
            if should_truncate_choice:
                truncation_in = random.randint(0, 2)
                # For x-axis
                if truncation_in in [0, 2]:
                    truncate_percent_x = random.randint(10, 35)
                    is_truncate_left = random.choice([True, False])
                    if is_truncate_left:
                        card_start_x = floor(card_file_shape[0] * truncate_percent_x / 100.0)
                        card_end_x = card_file_shape[0]
                        background_start_x = 0 + start_index_x
                        background_end_x = card_end_x - card_start_x + start_index_x
                    else:
                        card_start_x = 0 
                        card_end_x = ceil(card_file_shape[0] * (100.0 - truncate_percent_x) / 100.0)
                        background_start_x = end_index_x - (card_end_x - card_start_x)
                        background_end_x = end_index_x
                else:
                    card_start_x, card_end_x = 0, card_file_shape[0]
                    background_start_x = random.randint(start_index_x, end_index_x - card_file_shape[0])
                    background_end_x = background_start_x + card_file_shape[0]
                    truncate_percent_x = 0
                
                # For y-axis
                if truncation_in in [1, 2]:
                    truncate_percent_y = random.randint(10, 35)
                    is_truncate_top = random.choice([True, False])
                    if is_truncate_top:
                        card_start_y = floor(card_file_shape[1] * truncate_percent_y / 100.0)
                        card_end_y = card_file_shape[1]
                        background_start_y = 0 + start_index_y
                        background_end_y = card_end_y - card_start_y + start_index_y
                    else:
                        card_start_y = 0 
                        card_end_y = ceil(card_file_shape[1] * (100.0 - truncate_percent_y) / 100.0)
                        background_start_y = end_index_y - (card_end_y - card_start_y)
                        background_end_y = end_index_y
                else:
                    card_start_y, card_end_y = 0, card_file_shape[1]
                    background_start_y = random.randint(start_index_y, end_index_y - card_file_shape[1])
                    background_end_y = background_start_y + card_file_shape[1]
                    truncate_percent_y = 0
                    
                is_difficult = (truncate_percent_x + truncate_percent_y) > 40
                is_occluded = (truncate_percent_x + truncate_percent_y) > 55
                is_truncated = True
            else:
                card_start_x, card_end_x = 0, card_file_shape[0]
                background_start_x = random.randint(start_index_x, end_index_x - card_file_shape[0])
                background_end_x = background_start_x + card_file_shape[0]
                
                card_start_y, card_end_y = 0, card_file_shape[1]
                background_start_y = random.randint(start_index_y, end_index_y - card_file_shape[1])
                background_end_y = background_start_y + card_file_shape[1]
                
                is_difficult = False
                is_truncated = False
                is_occluded = False

            background_img[background_start_x: background_end_x, background_start_y: background_end_y, :] = card_files_op[card_index_at,\
                card_start_x : card_end_x, card_start_y : card_end_y, :]       
            boxes.append((background_start_y, background_start_x, background_end_y, background_end_x))
            image_card_labels.append(card_labels[card_index_at])
            is_difficult_array.append(is_difficult)
            is_truncated_array.append(is_truncated)
            is_occluded_array.append(is_occluded)
            
            card_index_at += 1
        
        noise_type = random.randint(0, 2)   # 0: None, 1: Gaussian, 2: Pepper
        if noise_type == 1:
            background_img = add_gaussian_noise(background_img)
        elif noise_type == 2:
            background_img = add_salt_pepper_noise(background_img)
            
        save_location = '{}/{:06d}.png'.format(save_img_folder, current_save_index)
        mpimg.imsave(save_location, background_img)
        
        img_full_path = os.path.join(cwd, save_location)
        write_xml_file(boxes, image_card_labels, background_img.shape, img_full_path, save_xml_folder,
                      is_truncated_array, is_difficult_array, is_occluded_array)
        
        current_save_index += 1
    return current_background_index, no_of_files_array, card_labels

def write_label_file_entries(label_entries, no_of_entries, save_folder, folder_type):
    file_path = '{}/{}_{}.txt'.format(save_folder, LABEL, folder_type)
    label_file = []
    label_index = 0
    with open(file_path, 'a') as file:
        for entries in no_of_entries:
            for entry_no in range(entries):
                file.write('{}'.format(label_entries[label_index]))
                if entry_no == entries - 1:
                    file.write('\n')
                else:
                    file.write(',')
                label_index += 1