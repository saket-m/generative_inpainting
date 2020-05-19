import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import os
import shutil


from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--width', default=400, type=int,
                    help='width of the video')
parser.add_argument('--height', default=400, type=int,
                    help='height of the video')

if __name__ == "__main__":
    ## python sony_test_2.py --checkpoint_dir model_logs/release_places2_256_deepfill_v2
    from tensorflow.python.client import device_lib
    print('######################', device_lib.list_local_devices())

    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()
    checkpoint_dir = args.checkpoint_dir

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

    ## 2. load the tensorflow model and create a session
    ## load pretrained model
    model = InpaintCAModel()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    input_image = tf.placeholder(shape = (1, args.height, args.width * 2, 3), dtype=tf.float32, name='fdfd')
    output = model.build_server_graph(FLAGS, input_image)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')
    

    ## 3. iterate over the videos and extract frames and based on the position
    ## map crear the logo and re paint the frame
    print(1111111111, os.listdir(VIDEO_DIR))
    for video_name in os.listdir(VIDEO_DIR):
        video_path = os.path.join(VIDEO_DIR, video_name)
        mask_path = os.path.join(MASK_DIR, video_name).replace('mp4', 'png').replace('MP4', 'png')
        print(2222222222, mask_path, video_path)

        shutil.rmtree(TEMP_INPUT_DIR)
        shutil.rmtree(TEMP_OUTPUT_DIR)
        os.mkdir(TEMP_INPUT_DIR)
        os.mkdir(TEMP_OUTPUT_DIR)
        os.system(f'ffmpeg -i {video_path} {TEMP_INPUT_DIR}/%05d.png')

        i = 0
        for image_name in os.listdir(TEMP_INPUT_DIR):
            image_path = os.path.join(TEMP_INPUT_DIR, image_name)
            output_path = os.path.join(TEMP_OUTPUT_DIR, image_name)

            ## 4. load the image and mask for inference
            image = cv2.imread(image_path)
            image_orig = image.copy()
            mask = cv2.imread(mask_path)
            mask_orig = mask.copy()

            ## crop the image
            image = image[:400, image.shape[1]-400:, :]
            cv2.imwrite('a.png', image)
            mask = mask[:400, mask.shape[1]-400:, :]
            cv2.imwrite('b.png', mask)

            # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
            print(image.shape, mask.shape)
            assert image.shape == mask.shape

            h, w, _ = image.shape
            grid = 8
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid, :]
            print(i, 'Shape of image: {}'.format(image.shape))
            i += 1

            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image_mask = np.concatenate([image, mask], axis=2)           
            
            result = sess.run(output, feed_dict={input_image:input_image_mask})
            # cv2.imwrite(output_path,result[0][:, :, ::-1])
            result = result[0][:, :, ::-1]
            image_orig[mask_orig > 0] = result[mask[0] > 0]
            cv2.imwrite(output_path, image_orig)

            # test(image_path, mask_path, checkpoint_dir, output_path)
            
        ## 4. stitch the repainted frames and store them in a output dir
        os.system(f'ffmpeg -i {TEMP_OUTPUT_DIR}/%05d.png {OUTPUT_DIR}/{video_name}')
    
