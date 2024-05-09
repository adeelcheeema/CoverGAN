import json
import os
from os import path
# import cv2 as cv
# import numpy as np

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from model import get_model, json_to_img,json_to_hd_img
model = get_model()

def run():

    # imagemain = cv2.imread('/Users/adeelcheema/Desktop/Image Generation/scene_generation/scripts/gui/5.jpg')
    # image = imagemain[:,256:,:]
    # image2 = imagemain[:,:256,:]
    #
    # for ii in range(0, 255):
    #     for jj in range(0, 255):
    #             if ((np.sum(image[ii][jj]) / 3) > 0):
    #                 image2[ii][jj] = image[ii][jj]
    #
    # cv2.imshow(image2)
    #
    # return

    filename = 'E:\AdeelCoverGAN\Image Generation\scene_generation\scripts\gui\scene_graphs.json'
    listObj = []

    # Check if file exists
    if path.isfile(filename) is False:
        raise Exception("File not found")

    # Read JSON file
    with open(filename) as fp:
        listObj = json.load(fp)
    indd = 0
    for i in listObj:
        json_to_hd_img(i, model)
        indd=indd + 1
        print("=== Generated " + str(indd) + " / " + str(len(listObj)) + " ==>")
    return

run()
