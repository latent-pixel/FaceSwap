import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
#import matplotlib.pyplot as plt
import argparse

from Phase2.api import PRN
from Phase2.api_2face import PRN_2face
from Phase2.utils.render import render_texture
import cv2


def fswap(prn, pos, ref_pos, ref_image, image):
    [h, w, _] = image.shape
        #-- 1. 3d reconstruction -> get texture. 
    
    vertices = prn.get_vertices(pos)
    image = image/255.
    texture = cv2.remap(image, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

    #-- 2. Texture Editing
    
    ref_image = ref_image/255.
    ref_texture = cv2.remap(ref_image, ref_pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    ref_vertices = prn.get_vertices(ref_pos)
    new_texture = ref_texture

    #-- 3. remap to input image.(render)
    vis_colors = np.ones((vertices.shape[0], 1))
    face_mask = render_texture(vertices.T, vis_colors.T, prn.triangles.T, h, w, c = 1)
    face_mask = np.squeeze(face_mask > 0).astype(np.float32)
    
    new_colors = prn.get_colors_from_texture(new_texture)
    new_image = render_texture(vertices.T, new_colors.T, prn.triangles.T, h, w, c = 3)
    new_image = image*(1 - face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]

    # Possion Editing for blending image
    vis_ind = np.argwhere(face_mask>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    output = cv2.seamlessClone((new_image*255).astype(np.uint8), (image*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)
    
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--InputFilePath', default='my_test_imgs/', help='Folder in which the input files are located')
    parser.add_argument('--InputFileName', default='Test2.mp4', help='Name of the input file')
    parser.add_argument('--RefFileName', default='None', help='Reference image to be used for swapping over a frame (if any)')
    parser.add_argument('--SaveFileName', default='result.avi', help='Name of the output file')

    args = parser.parse_args()
    InputFilePath = args.InputFilePath
    InputFileName = args.InputFileName
    RefFileName = args.RefFileName
    FileSavePath = 'results/' + args.SaveFileName

    InputPath = InputFilePath + InputFileName
    # print(InputPath)
    RefPath = InputFilePath + RefFileName
    SavePath = InputFilePath + FileSavePath

    ref_img = cv2.imread(RefPath)
    mode = 0
    if ref_img is None:
        mode = 1

    cap = cv2.VideoCapture(InputPath)  # VideoCapture object
    #disp_duration = 10
    if not cap.isOpened():
        print("Error opening video stream or file!")
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    result = cv2.VideoWriter(SavePath, cv2.VideoWriter_fourcc(*'DIVX'), 30, (frame_width, frame_height)) 
    count = 0
    all_frames = list()

    if mode == 0:
        prn = PRN(is_dlib = True)
        old_pos = None
        ref_pos = prn.process(ref_img)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                count += 1
                print("Evaluating frame ", count)
                all_frames.append(frame)
                new_pos = prn.process(frame)
                if new_pos is not None: 
                    blended_frame = fswap(prn, new_pos, ref_pos, ref_img, frame)
                    cv2.imshow('frame', blended_frame)
                else:
                    if old_pos is not None:
                        new_pos = old_pos
                    else:
                        blended_frame = frame
                result.write(blended_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        result.release() 
        cv2.destroyAllWindows()
    else:
        # Swapping in the scenario when there are two faces in the video
        prn = PRN_2face(is_dlib = True)
        # kernel = np.array([[0, -1, 0],
        #                 [-1, 5,-1],
        #                 [0, -1, 0]])
        old_pos = None
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                count += 1
                print("Evaluating frame ", count)
                frame[frame > 255] = 255
                # frame_sharp = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
                # if count >= 1 and count <= 10:
                all_frames.append(frame)
                new_pos = prn.process(frame)
                if new_pos is None or len(new_pos) < 2:
                    new_pos = old_pos
                if len(new_pos) == 2:
                    old_pos = new_pos
                    pos1, pos2 = new_pos[0], new_pos[1]
                    blended_frame = fswap(prn, pos1, pos2, frame, frame)
                    blended_frame_final = fswap(prn, pos2, pos1, frame, blended_frame)
                else:
                    print("No faces here!")
                    blended_frame_final = frame
                cv2.imshow('frame', blended_frame_final)
                result.write(blended_frame_final)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        result.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()