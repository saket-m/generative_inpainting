import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import os
import shutil


from inpaint_model import InpaintCAModel


if __name__ == "__main__":
    '''
    this script takes a video and removes logo from it, the logo
    should be present in a fixed  location across all the frames.
    '''
    ## 1. set the input video dir, it may contain multiple videos and 
    ## the frame position map
    VIDEO_DIR = './examples/sony/input_videos'
    MASK_DIR = './examples/sony/mask'
    OUTPUT_DIR = './examples/sony/output_videos'
    TEMP_DIR = './examples/sony/temp'
    TEMP_INPUT_DIR = './examples/sony/temp/input'
    TEMP_OUTPUT_DIR = './examples/sony/temp/output'

    if not os.path.exists(VIDEO_DIR):
        print('E :: inout video dir does not exist.')
        exit()
    if not os.path.isdir(MASK_DIR):
        print('E :: inout mask dir does not exist.')
        exit()
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    if not os.path.isdir(TEMP_DIR):
        os.mkdir(TEMP_DIR)
    if not os.path.isdir(TEMP_INPUT_DIR):
        os.mkdir(TEMP_INPUT_DIR)
    if not os.path.isdir(TEMP_OUTPUT_DIR):
        os.mkdir(TEMP_OUTPUT_DIR)


    ## 3. iterate over the videos and extract frames and based on the position
    ## map crear the logo and re paint the frame
    print(1, os.listdir(VIDEO_DIR))
    for video_name in os.listdir(VIDEO_DIR):
        video_path = os.path.join(VIDEO_DIR, video_name)
        mask_path = os.path.join(MASK_DIR, video_name).replace('mp4', 'png').replace('MP4', 'png')
        print(2, mask_path, video_path)

        ''' shutil.rmtree(TEMP_INPUT_DIR)
        shutil.rmtree(TEMP_OUTPUT_DIR)
        os.mkdir(TEMP_INPUT_DIR)
        os.mkdir(TEMP_OUTPUT_DIR)
        os.system(f'ffmpeg -i {video_path} {TEMP_INPUT_DIR}/%05d.png')
        '''

        i = 0
        for image_name in os.listdir(TEMP_INPUT_DIR):
            image_path = os.path.join(TEMP_INPUT_DIR, image_name)
            output_path = os.path.join(TEMP_OUTPUT_DIR, image_name)

            ## 4. load the image and mask for inference
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)
            mask = mask.astype('uint8')
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
            print(image.shape, mask.shape)

            h, w, _ = image.shape
            i += 1

            print(np.unique(mask))
            output = cv2.inpaint(image, mask, 3, flags=cv2.INPAINT_NS)
            cv2.imwrite(output_path, output)
            print(f'{i} images completed.')

            # test(image_path, mask_path, checkpoint_dir, output_path)
            
        ## 4. stitch the repainted frames and store them in a output dir
        os.system(f'ffmpeg -i {TEMP_OUTPUT_DIR}/%05d.png -crf 0 {OUTPUT_DIR}/cv2_{video_name}')
    
