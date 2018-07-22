import argparse
from math import ceil, floor, pi, sqrt
from tqdm import tqdm
import numpy as npBACKGROUND
import os
from sklearn.utils import shuffle
import shutil
import random

from data_utils.operations import tf_generate_images, write_label_file_entries, instantiate_global_variables
from data_utils.constants import XML_FOLDER, GENERATED_DATA, TRAIN_FOLDER, VAL_FOLDER, TEST_FOLDER, LABEL
from data_utils.constants import CARD_HEIGHT, CARD_WIDTH, MIN_CARDS, MAX_CARDS
from data_utils.constants import BACKGROUND_TOTAL_FILES, BORDER_WHITE_AREA
from data_utils.constants import TOTAL_TRAIN_IMAGES, TOTAL_VALIDATION_IMAGES

BATCH_SIZE = 16

def get_filenames_and_labels(card_path):
    card_file_names = os.listdir(card_path)
    card_file_names.sort()
    card_file_names = ['{}/{}'.format(card_path, card_file) for card_file in card_file_names]

    labels = list(range(len(card_file_names)))
    return card_file_names, labels

def generate_image_pipeline(X_files, y_data, save_folder, folder_type, bg_img_folder,
							start_background_index, total_base_images,
                            scales = [0.80, 0.81, 0.82, 0.83, 0.84, 0.85],
                            angles = [0], angle_repeat_ratio = [1]):
    # In case, the number of cards are very less, to keep the logic simple, 
    # set the batch size not larger than number of cards.
    effective_batch_size = min(BATCH_SIZE, len(y_data))

    # Folder for saving generated images.
    save_img_folder = '{}/{}_{}'.format(save_folder, GENERATED_DATA, folder_type)
    shutil.rmtree(save_img_folder, ignore_errors = True)
    os.mkdir(save_img_folder)
    # Folder for saving annotation XML files.
    save_xml_folder = '{}/{}_{}'.format(save_folder, XML_FOLDER, folder_type)
    shutil.rmtree(save_xml_folder, ignore_errors = True)
    os.mkdir(save_xml_folder)
    # File for saving labels.
    file_name = '{}/{}_{}.txt'.format(save_folder, LABEL, folder_type)
    if os.path.exists(file_name):
        os.unlink(file_name)
        
    # Counter indexes.
    save_index = 0  # Index for maintaining saved file number of newly generated file.
    background_index = start_background_index # Index for maintaining at which file index of bg_img is at currently. Loops after finishing.
    scale_index = 0  # Index for at which scale position of card image. Loops after finishing.
    data_index = 0 # Index for maintaining at which card image is currently at. Loops after finishing.
    
    data_samples = len(y_data)
    # Calculate the number of images to generate for each angle.
    angle_images = [ceil(total_base_images * ratio) for ratio in angle_repeat_ratio]
    total_images = sum(angle_images)
    with tqdm(total = total_images) as pbar:
        instantiate_global_variables()
        
        # Generate total images needed at each angle.
        for angle_at, images_at_angle in zip(angles, angle_images):
            save_image_at = 0
            while save_image_at < images_at_angle:
                # Get the scale index.
                if scale_index == len(scales):
                    scale_index = 0
                scale_at = scales[scale_index]
                scale_index += 1
                
                if data_index >= data_samples:
                    data_index = 0
                    
                no_of_files_array = []
                # Keep the ability of putting multiple card files in one image only if scaling is below 0.5. 
                if scale_at <= 0.5:
                    for batch_counter in range(min(images_at_angle - save_image_at, effective_batch_size)):
                        files_to_pick = random.randint(MIN_CARDS, MAX_CARDS)
                        no_of_files_array.append(files_to_pick)
                else:
                    no_of_files_array = [MIN_CARDS] * min(images_at_angle - save_image_at, effective_batch_size)
                no_of_files = sum(no_of_files_array)
                
                # Collect the needed number of card files. 
                if data_index + no_of_files > data_samples:
                    # This condition deals with in case the looping of card files array has to be done.
                    batch_X_files = X_files[data_index: ]
                    batch_y_data = y_data[data_index: ]
                    data_index = no_of_files - len(batch_y_data)
                    batch_X_files.extend(X_files[: data_index])
                    batch_y_data = np.concatenate((batch_y_data, y_data[: data_index]))
                    # If the data is not filled still.
                    if len(batch_y_data) != no_of_files:
                        data_index = no_of_files - len(batch_y_data)
                        batch_X_files.extend(X_files[: data_index])
                        batch_y_data = np.concatenate((batch_y_data, y_data[: data_index]))
                else:
                    batch_X_files = X_files[data_index: data_index + no_of_files]
                    batch_y_data = y_data[data_index: data_index + no_of_files]
                    data_index += no_of_files
                    
                # Some check to see if required number of cards files are collected.
                # Ideally, the assert condition should never fail.
                assert no_of_files == len(batch_X_files), 'Length mismatch in data files'
                assert no_of_files == len(batch_y_data), 'Length mismatch in label array'

                # As there are large number of parameters to pass, pass it in a dictionary.
                parameter_dict = {'scale_at': scale_at, 
                                    'angle_at': angle_at,
                                    'background_index_at': background_index,
                                    'save_index': save_index,
                                    'raw_card_size': (CARD_HEIGHT, CARD_WIDTH),
                                    'no_of_files_array': no_of_files_array,
                                    'border_area': BORDER_WHITE_AREA,
                                    'bg_total_files': BACKGROUND_TOTAL_FILES}

                # Generate the batch of images.
                background_index, no_of_files_array, batch_y_data = tf_generate_images(batch_X_files, batch_y_data,
                                                                        bg_img_folder, save_img_folder,
                                                                        save_xml_folder, parameter_dict)

                write_label_file_entries(batch_y_data, no_of_files_array, save_folder, folder_type)
                save_index += len(no_of_files_array)
                save_image_at += len(no_of_files_array)
                pbar.update(len(no_of_files_array))


def parse_args():
    parser = argparse.ArgumentParser(description = 'Resize data and prepare labels')
    parser.add_argument('card_folder', help = 'Where cards are present', type = str)
    parser.add_argument('bg_img_folder', help = 'Where background images are present', type = str)
    parser.add_argument('--save-folder', dest = 'save_folder', help = 'Where the generated files are to be stored', 
            default = os.path.join(os.getcwd(), 'input_data'), type = str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
	args = parse_args()

	label_file_names, labels = get_filenames_and_labels(args.card_folder)
	label_file_names, labels = shuffle(label_file_names, labels)

	generate_image_pipeline(label_file_names, labels, args.save_folder, TRAIN_FOLDER, args.bg_img_folder, 
							1, total_base_images = TOTAL_TRAIN_IMAGES)
	generate_image_pipeline(label_file_names, labels, args.save_folder, VAL_FOLDER, args.bg_img_folder, 
							190000, total_base_images = TOTAL_VALIDATION_IMAGES)