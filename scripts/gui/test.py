#!/usr/bin/env python3
# from email.quoprimime import unquote
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import unquote



# import sys
# sys.path.append('../')

from model import get_model, json_to_img


model = get_model()


def run():
    temp = path.split('/get_data?data=')[1]
    img_path, layout_path = json_to_img(path.split('/get_data?data=')[1], model)
    paths = json.dumps({'img_pred': img_path, 'layout_pred': layout_path}))
run()
