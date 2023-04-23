import onnxruntime
import cv2
import numpy as np
from .utils import CTCLabelConverter
import math
import os
import gdown

character_list = "!\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`{|}~\xD7\xE0\xE1\xE9\u0104\u0E01\u0E02\u0E03\u0E04\u0E05\u0E06\u0E07\u0E08\u0E09\u0E0A\u0E0B\u0E0C\u0E0D\u0E0E\u0E0F\u0E10\u0E11\u0E12\u0E13\u0E14\u0E15\u0E16\u0E17\u0E18\u0E19\u0E1A\u0E1B\u0E1C\u0E1D\u0E1E\u0E1F\u0E20\u0E21\u0E22\u0E23\u0E24\u0E25\u0E26\u0E27\u0E28\u0E29\u0E2A\u0E2B\u0E2C\u0E2D\u0E2E\u0E2F\u0E30\u0E31\u0E32\u0E33\u0E34\u0E35\u0E36\u0E37\u0E38\u0E39\u0E3A\u0E3F\u0E40\u0E41\u0E42\u0E43\u0E44\u0E45\u0E46\u0E47\u0E48\u0E49\u0E4A\u0E4B\u0E4C\u0E4D\u0E50\u0E51\u0E52\u0E53\u0E54\u0E55\u0E56\u0E57\u0E58\u0E59\u2013\u2014\u2018\u2019\u201C\u201D\u2022\u2026\u2713\uFEFF"


def get_file_path():
    return os.path.dirname(os.path.realpath(__file__))


def download_model():
    if not os.path.exists(get_file_path() + '/crnn.onnx'):
        url = 'https://drive.google.com/file/d/1r5-X91uRKrdDSpqygucc_heVsYtt6tVX/view?usp=share_link'
        output = get_file_path() + '/crnn.onnx'
        gdown.download(url, output, quiet=False, fuzzy=True)


def init_converter():
    converter = CTCLabelConverter(character_list)
    return converter


def init_session(provider='CPUExecutionProvider'):
    ort_session = onnxruntime.InferenceSession(
        get_file_path() + '/crnn.onnx', providers=[provider])
    return ort_session


def normalize_pad(img, max_size):
    img = np.subtract(img, 0.5)
    img = np.multiply(img, 2.0)
    h, w = img.shape
    pad_img = np.zeros(max_size, dtype=np.float32)
    pad_img[:, :w] = img
    if max_size[1] != w:
        pad_img[:, w:] = img[:, w-1].reshape(h, 1)
    return pad_img


def resize(img, size):
    ratio = img.shape[1] / img.shape[0]
    resized_w = 0
    if math.ceil(size[0] * ratio) > size[1]:
        resized_w = size[1]
    else:
        resized_w = math.ceil(size[0] * ratio)
    resized_img = cv2.resize(img, (resized_w, size[0]))
    return resized_img


def preprocess(img, size=(64, 600)):
    img = img / 255.0
    img = resize(img, size)
    img = normalize_pad(np.array(img), size)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return img


def predict(img, ort_session, converter):
    img = preprocess(img)
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    pred = ort_outs[0]
    pred = converter.decode_beamsearch(pred, beamWidth=1)
    return pred


def model_recognition(imgs):
    download_model()
    ort_session = init_session()
    converter = init_converter()
    output = []
    for img in imgs:
        # img = cv2.imread('2.png', cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pred = predict(img, ort_session, converter)
        output.append(pred[0])
        # print(pred)
    return output


if __name__ == '__main__':
    model_recognition([])
