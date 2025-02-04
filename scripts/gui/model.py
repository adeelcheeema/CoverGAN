import argparse
import json
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from imageio import imwrite

from PIL import Image
from PIL import ImageOps
import cv2 as cv

import random
import textwrap
from sklearn.cluster import MiniBatchKMeans
from skimage import io
from skimage.transform import rescale

import sys

print(sys.path)

# import scene_generation.vis as vis
from scene_generation import vis

from scene_generation.data.utils import imagenet_deprocess_batch
from scene_generation.model import Model

from Diffusion import gen_image

import PIL
import requests
import torch

from io import BytesIO

from diffusers import StableDiffusionInpaintPipeline
import PIL

import torch

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    variant="fp16",
    torch_dtype=torch.float16,
)

pipe = pipe.to("cuda")

# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint', required=True, default='../../models/128/checkpoint_with_model.pt')
# parser.add_argument('--output_dir', default='outputs')
# parser.add_argument('--draw_scene_graphs', type=int, default=0)
# parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])
# args = parser.parse_args()

_checkpoint = '../../models/256/checkpoint_with_model.pt'
_output_dir = 'outputs'
_draw_scene_graphs = 0
_device = 'cpu'


def get_model():
    if not os.path.isfile(_checkpoint):
        print('ERROR: Checkpoint file "%s" not found' % _checkpoint)
        print('Maybe you forgot to download pretraind models? Try running:')
        print('bash scripts/download_models.sh')
        return

    output_dir = os.path.join('scripts', 'gui', 'images', _output_dir)
    if not os.path.isdir(output_dir):
        print('Output directory "%s" does not exist; creating it' % _output_dir)
        os.makedirs(output_dir)

    if _device == 'cpu':
        device = torch.device('cpu')
    elif _device == 'gpu':
        device = torch.device('cuda:0')
        if not torch.cuda.is_available():
            print('WARNING: CUDA not available; falling back to CPU')
            device = torch.device('cpu')

    # Load the model, with a bit of care in case there are no GPUs
    map_location = 'cpu' if device == torch.device('cpu') else None
    checkpoint = torch.load(_checkpoint, map_location=map_location)
    dirname = os.path.dirname(_checkpoint)
    features_path = os.path.join(dirname, 'features_clustered_100.npy')
    features_path_one = os.path.join(dirname, 'features_clustered_001.npy')
    features = np.load(features_path, allow_pickle=True).item()
    features_one = np.load(features_path_one, allow_pickle=True).item()
    model = Model(**checkpoint['model_kwargs'])
    model_state = checkpoint['model_state']
    model.load_state_dict(model_state)
    model.features = features
    model.features_one = features_one
    model.colors = torch.randint(0, 256, [172, 3]).float()
    model.colors[0, :] = 256
    model.eval()
    model.to(device)
    return model


def compare_image(base_img, test_img):
    base = base_img
    testA = test_img

    hsv_base = cv.cvtColor(base, cv.COLOR_BGR2HSV)
    hsv_testA = cv.cvtColor(testA, cv.COLOR_BGR2HSV)

    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges
    channels = [0, 1]

    hist_base = cv.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    hist_testA = cv.calcHist([hsv_testA], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_testA, hist_testA, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    compare_method = cv.HISTCMP_CORREL
    base_testA = cv.compareHist(hist_base, hist_testA, compare_method)

    # print('base_test SimilarityA = ', base_testA)
    return base_testA


def json_to_img(scene_graph, model):
    colors = {
        "bear": [230, 25, 75],
        "cat": [60, 180, 75],
        "cow": [255, 225, 25],
        "dog": [0, 130, 200],
        "elephant": [245, 130, 48],
        "giraffe": [145, 30, 180],
        "horse": [70, 240, 240],
        "sheep": [240, 50, 230],
        "zebra": [210, 245, 60],
        "sky-other": [250, 190, 212],
        "clouds": [0, 128, 128],
        "mountain": [220, 190, 255],
        "sky": [170, 110, 40],
        "hill": [255, 250, 200],
        "grass": [128, 0, 0],
        "river": [170, 195, 255],
        "sea": [128, 128, 0]}

    output_dir = _output_dir
    # scene_graphs = json_to_scene_graph(scene_graph)
    scene_graphs = [scene_graph]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    # Run the model forward
    with torch.no_grad():
        (imgs, boxes_pred, masks_pred, layout, layout_pred, _), objs = model.forward_json(scene_graphs)
    imgs = imagenet_deprocess_batch(imgs)

    # Save the generated image
    for i in range(imgs.shape[0]):
        img_np = imgs[i].numpy().transpose(1, 2, 0).astype('uint8')
        # img_path = os.path.join('outputs', 'lowres', 'img{}.png'.format(current_time))
        # imwrite(img_path, img_np)
        path = 'E:\AdeelCoverGAN\Image Generation\scene_generation\/outputs/lowres/' + current_time + '_' + \
               scene_graphs[0]['image_id'] + '.png'
        cv.imwrite(path, img_np)
        return_img_path = os.path.join('outputs', 'lowres', 'img{}.png'.format(current_time))

    # Save the generated layout image

    for i in range(imgs.shape[0]):
        img_layout_np = one_hot_to_rgb(layout_pred[:, :172, :, :], model.colors)[0].numpy().transpose(1, 2, 0).astype(
            'uint8')

        obj_colors = []
        g_img_list = img_layout_np.tolist()
        c = 0
        ## MASK COLOR CHANGING

        main_mask = np.zeros((512, 512), dtype=np.uint8)
        generated_images = []
        boundingBoxes = []
        for object in scene_graphs[i]['objects'][:-1]:
            new_color = torch.cat((torch.div(torch.FloatTensor(colors[object]), 255), torch.ones(1)))
            obj_colors.append(new_color)
            bbox = (boxes_pred[c].numpy() * 255).astype(int)

            if object in ["elephant", "zebra", "sheep"]:
                scale_width = 512 / 256
                scale_height = 512 / 256

                bbox_scaled = np.array([
                    bbox[0] * scale_width,  # x1 scaled
                    bbox[1] * scale_height,  # y1 scaled
                    bbox[2] * scale_width,  # x2 scaled
                    bbox[3] * scale_height  # y2 scaled
                ]).astype(int)

                x, y, w, h = bbox_scaled
                w, h = min(x + w, 512), min(y + h, 512)
                object_mask = np.zeros((512, 512), dtype=np.uint8)
                cv.rectangle(object_mask, (x, y), (w, h), color=255,
                             thickness=cv.FILLED)  # Use 255 for white color on binary image

                # bounding box erosion
                x_e, y_e = x + 30, y + 30
                w_e, h_e = w - 30, h - 30
                cv.rectangle(main_mask, (x_e, y_e), (w_e, h_e), color=255, thickness=cv.FILLED)

                main_image = Image.new('RGB', (512, 512), (0, 0, 0))
                prompt = "Create a highly detailed digital painting of a " + object + " in a realistic style. The " + object + " should be in a dynamic pose. The " + object + " must be in 4K resolution";
                hd_images = gen_image(prompt, main_image, object_mask)
                generated_images.append(hd_images)
                boundingBoxes.append([x, y, w, h])

            x, y, w, h = bbox

            for ii in range(y, h):
                for jj in range(x, w):
                    if ii < 256 and jj < 256:
                        if ((np.sum(img_layout_np[ii][jj]) / 3) < 250):
                            # if object in ["elephant", "zebra", "sheep"]:
                            g_img_list[ii][jj] = colors[object]
                            # else:
                            #     g_img_list[ii][jj] = [255,255,255]
            c = c + 1
        #
        # joined_image = PIL.Image.new('RGB', (512, 512), (0, 0, 0))
        # x, y, w, h = boundingBoxes[0]
        # joined_image.paste(generated_images[0][0].crop((x, y, w, h)), (x, y))
        # x, y, w, h = boundingBoxes[1]
        # joined_image.paste(generated_images[1][0].crop((x, y, w, h)), (x, y))

        results = []
        process_combinations(generated_images, boundingBoxes, results=results)

        prompt_fin = "tropical jungle or desert or mountian or medow or trees or zoo, realistic, 4k"
        object_mask_fin = ImageOps.invert(Image.fromarray(main_mask))
        title = "The " + " ".join(scene_graphs[i]['objects'][:-1] + ["and"] * (len(scene_graphs[i]['objects']) > 1) + scene_graphs[i]['objects'][-1:])

        for gen_img in results:
            hd_images = gen_image(prompt_fin, gen_img, object_mask_fin)
            for iii, img in enumerate(hd_images):
                path = 'E:/AdeelCoverGAN/Image Generation/scene_generation/outputs/hd_images/' + current_time + '_' + \
                       scene_graphs[i]['image_id'] + '_' + iii + '.png'



        img_layout_new = np.asarray(g_img_list).astype('uint8')
        changed_mask = np.asarray(g_img_list).astype('uint8')

        # for obj in objs[:-1]:
        #     new_color = torch.cat([model.colors[obj] / 256, torch.ones(1)])
        #     obj_colors.append(new_color)

        c = 0
        ## MASK IMPROVEMENT

        g_img_blk = img_layout_new
        g_img_blk[np.where((g_img_blk == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
        g_img_blk[np.where((g_img_blk == [254, 254, 254]).all(axis=2))] = [0, 0, 0]

        updated_mask = np.zeros(img_layout_new.shape, dtype='uint8', order='C')
        old_mask = np.zeros(img_layout_new.shape, dtype='uint8', order='C')
        prompt = ''
        for object in scene_graphs[i]['objects'][:-1]:
            if object in ["elephant", "zebra", "sheep"]:
                bbox = (boxes_pred[c].numpy() * 255).astype(int)

                x, y, w, h = bbox

                t_height = h - y
                t_weight = w - x

                if (t_height > t_weight):
                    t_weight = t_weight + (t_height - t_weight)
                elif (t_weight > t_weight):
                    t_height = t_height + (t_weight - t_height)

                r_height = y + t_height
                r_weight = x + t_weight

                ## Mask Improvement

                path = "E:/AdeelCoverGAN/Image Generation/scene_generation/scripts/gui/masks/" + object + "_old"

                base_mask = img_layout_new[y:y + t_height, x:x + t_weight]
                similarity = 0
                similar_path = ""
                for f in os.listdir(path):
                    comp = cv.imread(path + '/' + f)
                    res = compare_image(base_mask, comp)
                    if (res > similarity):
                        similarity = res
                        similar_path = path + '/' + f

                transform = T.Resize((t_height, t_weight))
                # im_r = Image.open(similar_path)
                im_r = cv.imread(similar_path)
                # comppp = cv.imread(similar_path)
                # resized_img = cv.resize(im_r, dsize=(t_height, t_weight), interpolation=cv.INTER_CUBIC)
                # resized_img = np.asarray(transform(im_r)).astype('uint8')
                resized_img = np.asarray(im_r).astype('uint8')

                resized_img = Image.fromarray(resized_img, mode='RGB')
                resized_img = np.asarray(transform(resized_img)).astype('uint8')

                img = resized_img[:, :, :3]
                offset = np.array((y, x))
                # background[offset[0]:offset[0] + img.shape[0], offset[1]:offset[1] + img.shape[1]] = img

                # cropped_x = [offset[0]:offset[0] + img.shape[0]]
                # cropped_y = offset[1]:offset[1] + img.shape[1]
                # for iii in range(cropped_x):
                #     for jjj in range(cropped_y):
                #         background[iii][jjj] = colors[object]

                indx = 0
                indy = 0

                t_r_height = h - y
                t_r_weight = w - x

                for ii in range(y, r_height):
                    indy = 0
                    for jj in range(x, r_weight):
                        if ii < 256 and jj < 256:
                            try:
                                p_sum = (np.sum(img[indx, indy]))
                                o_col_sum = (np.sum(colors[object]))

                                if ((o_col_sum) > (p_sum - 300) and (o_col_sum) < (p_sum + 300)):
                                    updated_mask[ii][jj] = colors[object]
                                    g_img_blk[ii][jj] = colors[object]

                                # if ((np.sum(img[indx,indy])) > 2 ):
                                #     updated_mask[ii][jj] = img[indx,indy]
                                #     g_img_blk[ii][jj] = img[indx, indy]
                                if ((np.sum(img_layout_np[ii, jj]) / 3) < 250):
                                    old_mask[ii][jj] = colors[object]
                                else:
                                    old_mask[ii][jj] = [0, 0, 0]
                                    g_img_blk[ii][jj] = [0, 0, 0]
                                # if ((np.sum(img_layout_np[ii, jj]) / 3) < 255):
                                #     g_img_blk[ii][jj] = img[indx,indy]
                                # else:
                                #     g_img_blk[ii][jj] = [0, 0, 0]
                            except NameError:
                                print(NameError)
                        indy = indy + 1
                    indx = indx + 1

                # imwrite(os.path.join('outputs', 'updated_mask', 'img{}.png'.format(current_time)), updated_mask)
                # imwrite(os.path.join('outputs', 'old_mask', 'img{}.png'.format(current_time)), old_mask)

            if c == 0:
                prompt = scene_graphs[i]['objects'][c]
            else:
                prompt = prompt + " and " + scene_graphs[i]['objects'][c]

            c = c + 1

            top_image = Image.fromarray(np.uint8(updated_mask)).resize((512, 512)).convert('L')
            fore_image = Image.fromarray(np.uint8(old_mask)).resize((512, 512)).convert('L')

            threshold_value = 10
            binary_top_image = top_image.point(lambda x: 0 if x < threshold_value else 255)
            binary_fore_image = fore_image.point(lambda x: 100 if x < threshold_value else 255)

            merged_image = Image.composite(binary_top_image, binary_fore_image, binary_top_image)
            back_img = Image.fromarray(np.uint8(img_np)).resize((512, 512))
            # back_img = Image.new('RGB', (512, 512))

            concatenated_image = np.concatenate([img_np, updated_mask, old_mask, ], axis=1)

            hd_imgages = []
            if object in ["elephant", "zebra", "sheep"]:
                hd_imgages = gen_image(prompt, back_img, merged_image)
                hd_imgages.insert(0, back_img)
                hd_imgages.insert(0, merged_image)
                image_grid(hd_imgages, 1, 5 + 2)
                concatenated_image = np.concatenate([np.array(hd_imgages[4]), np.array(back_img)], axis=1)
                aa = concatenated_image

            def save_images(images, directory):
                for idx, image in enumerate(images):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    image.save(f"{directory}/image_{idx}_{timestamp}.jpg")

            directory = "E:\AdeelCoverGAN\Image Generation\scene_generation\Diff_images"

            save_images(hd_imgages, directory)

        # Save the full figure...

        # kernel = np.ones((2, 2), np.uint8)
        #
        # img_dilation = cv.dilate(updated_mask, kernel, iterations=1)
        # img_erosion = cv.erode(img_dilation, kernel, iterations=1)

        path = 'E:/AdeelCoverGAN/Image Generation/scene_generation/outputs/updated_mask/' + current_time + '_' + \
               scene_graphs[0]['image_id'] + '.png'
        cv.imwrite(path, updated_mask)

        path = 'E:/AdeelCoverGAN/Image Generation/scene_generation/outputs/old_mask/' + current_time + '_' + \
               scene_graphs[0]['image_id'] + '.png'
        cv.imwrite(path, old_mask)

        # ###### Full Mask
        #
        # vis_mask = np.concatenate((img_layout_np,changed_mask,g_img_blk,old_mask,updated_mask), axis=1)
        # path = '/Users/adeelcheema/Desktop/Image Generation/scene_generation/output_masks/out_'+scene_graphs[0]['image_id']+'.png'
        # cv.imwrite(path, vis_mask)
        #
        # ###### Full Mask

        # img_layout_path = os.path.join('scripts', 'gui', 'images', output_dir, 'img_layout{}.png'.format(current_time))
        # # vis.add_boxes_to_layout(img_layout_new, scene_graphs[i]['objects'], boxes_pred, img_layout_path,
        # # #                         colors=obj_colors)
        # return_imreturn_img_layout_pathg_layout_path = os.path.join('images', output_dir, 'img_layout{}.png'.format(current_time))
        return_img_layout_path = ''

    # Draw and save the scene graph
    if _draw_scene_graphs:
        for i, sg in enumerate(scene_graphs):
            sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'])
            sg_img_path = os.path.join('scripts', 'gui', 'images', output_dir, 'sg{}.png'.format(current_time))
            imwrite(sg_img_path, sg_img)
            sg_img_path = os.path.join('images', output_dir, 'sg{}.png'.format(current_time))

    return return_img_path, return_img_layout_path


def json_to_hd_img(scene_graph, model):
    # scene_graphs = json_to_scene_graph(scene_graph)
    scene_graphs = [scene_graph]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    # Run the model forward
    with torch.no_grad():
        (imgs, boxes_pred, masks_pred, layout, layout_pred, _), objs = model.forward_json(scene_graphs)
    imgs = imagenet_deprocess_batch(imgs)

    # Save the generated image
    for i in range(imgs.shape[0]):
        img_np = imgs[i].numpy().transpose(1, 2, 0).astype('uint8')
        # img_path = os.path.join('outputs', 'lowres', 'img{}.png'.format(current_time))
        # imwrite(img_path, img_np)
        path = 'E:\AdeelCoverGAN\Image Generation\scene_generation\/outputs/lowres/' + current_time + '_' + \
               scene_graphs[0]['image_id'] + '.png'
        cv.imwrite(path, img_np)

    # Save the generated layout image

    for i in range(imgs.shape[0]):

        c = 0
        ## MASK COLOR CHANGING

        main_mask = np.zeros((512, 512), dtype=np.uint8)
        generated_images = []
        boundingBoxes = []
        for object in scene_graphs[i]['objects']:

            bbox = (boxes_pred[c].numpy() * 255).astype(int)

            if object in ["elephant", "zebra", "sheep"]:
                scale_width = 512 / 256
                scale_height = 512 / 256

                bbox_scaled = np.array([
                    bbox[0] * scale_width,  # x1 scaled
                    bbox[1] * scale_height,  # y1 scaled
                    bbox[2] * scale_width,  # x2 scaled
                    bbox[3] * scale_height  # y2 scaled
                ]).astype(int)

                x, y, w, h = bbox_scaled
                w, h = min(x + w, 512), min(y + h, 512)
                object_mask = np.zeros((512, 512), dtype=np.uint8)
                cv.rectangle(object_mask, (x, y), (w, h), color=255,
                             thickness=cv.FILLED)  # Use 255 for white color on binary image

                # bounding box erosion
                x_e, y_e = x + 20, y + 20
                w_e, h_e = w - 20, h - 20
                cv.rectangle(main_mask, (x_e, y_e), (w_e, h_e), color=255, thickness=cv.FILLED)

                main_image = Image.new('RGB', (512, 512), (0, 0, 0))
                prompt = "Create a highly detailed digital painting of a " + object + " in a realistic style. The " + object + " should be in a dynamic pose. The " + object + " must be in 4K resolution";
                hd_images = gen_image(prompt, main_image, object_mask)
                generated_images.append(hd_images)
                boundingBoxes.append([x, y, w, h])
            c = c + 1
            x, y, w, h = bbox

        results = []
        process_combinations(generated_images, boundingBoxes, results=results)

        back_str = [
            'tropical jungle', 'desert', 'mountain', 'meadow', 'trees',
            'forest', 'beach', 'river', 'rainforest',
            'city park'
        ]
        object_mask_fin = ImageOps.invert(Image.fromarray(main_mask))

        objects_list = scene_graphs[i]['objects'][:-1]

        if 'sky-other' in objects_list:
            objects_list.remove('sky-other')
        if 'grass' in objects_list:
            objects_list.remove('grass')

        title = 'The ' + ' '.join(objects_list)
        if len(objects_list) > 1:
            title = 'The ' + ' '.join(objects_list[:-1]) + ' and ' + objects_list[-1]
        # if(scene_graphs[i]['title']):
        #     title = scene_graphs[i]['title']
        for gen_img in results:
            random_environment = random.choice(back_str)
            prompt_fin =random_environment  + ", realistic, 4k"
            hd_images = gen_image(prompt_fin, gen_img, object_mask_fin)
            for iii, img in enumerate(hd_images):
                path_new = 'E:/AdeelCoverGAN/Image Generation/scene_generation/outputs/NEW_GEN/' + str(datetime.now().strftime('%b%d_%H-%M-%S')) + '_' + \
                       scene_graphs[i]['image_id'] + '_' + str(iii) + '.png'
                path_title = 'E:/AdeelCoverGAN/Image Generation/scene_generation/outputs/NEW_GEN_TITLE/' + str(
                    datetime.now().strftime('%b%d_%H-%M-%S')) + '_' + \
                           scene_graphs[i]['image_id'] + '_' + str(iii) + '.png'
                try:
                    coverImage(img, title, path_new , path_title)
                except:
                    img.save(path_new)

    return


def one_hot_to_rgb(one_hot, colors):
    one_hot_3d = torch.einsum('abcd,be->aecd', (one_hot.cpu(), colors.cpu()))
    one_hot_3d *= (255.0 / one_hot_3d.max())
    return one_hot_3d


def json_to_scene_graph(json_text):
    scene = json.loads(json_text)
    if len(scene) == 0:
        return []
    image_id = scene['image_id']
    scene = scene['objects']
    objects = [i['text'] for i in scene]
    relationships = []
    size = []
    location = []
    features = []
    for i in range(0, len(objects)):
        obj_s = scene[i]
        # Check for inside / surrounding

        sx0 = obj_s['left']
        sy0 = obj_s['top']
        sx1 = obj_s['width'] + sx0
        sy1 = obj_s['height'] + sy0

        margin = (obj_s['size'] + 1) / 10 / 2
        mean_x_s = 0.5 * (sx0 + sx1)
        mean_y_s = 0.5 * (sy0 + sy1)

        sx0 = max(0, mean_x_s - margin)
        sx1 = min(1, mean_x_s + margin)
        sy0 = max(0, mean_y_s - margin)
        sy1 = min(1, mean_y_s + margin)

        size.append(obj_s['size'])
        location.append(obj_s['location'])

        features.append(obj_s['feature'])
        if i == len(objects) - 1:
            continue

        obj_o = scene[i + 1]
        ox0 = obj_o['left']
        oy0 = obj_o['top']
        ox1 = obj_o['width'] + ox0
        oy1 = obj_o['height'] + oy0

        mean_x_o = 0.5 * (ox0 + ox1)
        mean_y_o = 0.5 * (oy0 + oy1)
        d_x = mean_x_s - mean_x_o
        d_y = mean_y_s - mean_y_o
        theta = math.atan2(d_y, d_x)

        margin = (obj_o['size'] + 1) / 10 / 2
        ox0 = max(0, mean_x_o - margin)
        ox1 = min(1, mean_x_o + margin)
        oy0 = max(0, mean_y_o - margin)
        oy1 = min(1, mean_y_o + margin)

        if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
            p = 'surrounding'
        elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
            p = 'inside'
        elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
            p = 'left of'
        elif -3 * math.pi / 4 <= theta < -math.pi / 4:
            p = 'above'
        elif -math.pi / 4 <= theta < math.pi / 4:
            p = 'right of'
        elif math.pi / 4 <= theta < 3 * math.pi / 4:
            p = 'below'
        relationships.append([i, p, i + 1])

    return [{'objects': objects, 'relationships': relationships, 'attributes': {'size': size, 'location': location},
             'features': features, 'image_id': image_id}]


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    plt.imshow(grid)
    plt.show()


def process_combinations(images, bounding_boxes, current_combo=[], current_bboxes=[], index=0, results=[]):
    if index == len(images):  # Base case: all lists have been processed
        # Process the current combination of images
        merged_image = merge_images(current_combo, current_bboxes)
        results.append(merged_image)
        return

    # Recursively process each image and bounding box in the current list
    for img_idx, img in enumerate(images[index]):
        process_combinations(images, bounding_boxes, current_combo + [img],
                             current_bboxes + [bounding_boxes[index]], index + 1, results)


def merge_images(images, bounding_boxes):
    # Create a new blank image with a fixed size
    joined_image = Image.new('RGB', (512, 512), (0, 0, 0))

    # Paste each image according to its bounding box
    for image, bbox in zip(images, bounding_boxes):
        x, y, w, h = bbox
        # Assume images and bounding boxes are correctly paired
        joined_image.paste(image.crop((x, y, w, h)), (x, y))

    return joined_image

def fast_dominant_colour(img_url, colours=5, scale=0.1):
  '''
  Faster method for web use that speeds up the sklearn variant.
  Also can use a scaling factor to improve the speed at cost of
  accuracy
  '''
  img = img_url
  if scale != 1.0:
    img = rescale(img, scale)
    img = img * 255
  img = img.reshape((-1, 3))

  cluster = MiniBatchKMeans(n_clusters=colours, n_init=3, max_iter=10, tol=0.001)
  cluster.fit(img)
  labels = cluster.labels_
  centroid = cluster.cluster_centers_

  percent = []
  _, counts = np.unique(labels, return_counts=True)
  for i in range(len(centroid)):
    j = counts[i]
    j = j / (len(labels))
    percent.append(j)

  indices = np.argsort(percent)[::-1]
  dominant = centroid[indices[0]]

  return dominant
def calculate_luminance(color):
    # Normalize RGB values to 0-1
    r, g, b = [x/255.0 for x in color]
    # Calculate luminance
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def coverImage(img, title, path, path_title):

    # image = cv.imread(img)
    image = np.array(img)
    cv.imwrite(path, cv.resize(image, (256, 256)))
    # image2 = imagemain[:,256:,:]
    # image = imagemain[:,:256,:]

    # for ii in range(0, 255):
    #   for jj in range(0, 255):
    #     if ((np.sum(image2[ii][jj]) / 3) > 20):
    #       image[ii][jj] = image2[ii][jj]
    coolor = (255, 255, 255)
    luminance = (0,0,0)

    try:
        coolor = fast_dominant_colour(image)
        luminance = calculate_luminance(coolor)
    except:
        print("error = "+path)

    inve_coolor = [220 - coolor[0], 220 - coolor[1], 220 - coolor[2]]
    border_col = inve_coolor[0] + inve_coolor[1] + inve_coolor[2] / 3

    inve_coolor = coolor
    border_col = luminance

    if (border_col > 127):
        border_col = [0, 0, 0]
    else:
        border_col = [255, 255, 255]

    cv2_fonts = [cv.FONT_HERSHEY_COMPLEX,
             # cv2.FONT_HERSHEY_COMPLEX_SMALL,
             cv.FONT_HERSHEY_DUPLEX,
             # cv2.FONT_HERSHEY_PLAIN,
             # cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
             cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
             cv.FONT_HERSHEY_SIMPLEX,
             cv.FONT_HERSHEY_TRIPLEX,
             cv.FONT_ITALIC];

    font = random.choice(cv2_fonts)

    wrapped_text = textwrap.wrap(title.upper(), width=16)
    x, y = 0, 0
    font_size = 1.2
    font_thickness = 3
    new_image = []

    for i, line in enumerate(wrapped_text):
        textsize = cv.getTextSize(line, font, font_size, font_thickness)[0]

        gap = textsize[1] + 10

        y = int(50) + i * gap
        x = int(50)

        new_image = cv.putText(image, line, (x, y), font,
                           font_size,
                           border_col,
                           font_thickness + 4,
                           lineType=cv.LINE_AA)

        new_image = cv.putText(image, line, (x, y), font,
                           font_size,
                           inve_coolor,
                           font_thickness,
                           lineType=cv.LINE_AA)

    cv.imwrite(path_title, cv.resize(new_image, (256, 256)))