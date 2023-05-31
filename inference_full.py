# Given a image, test full model output and time cost.
import onnx

import onnxruntime as ort

import numpy as np
import cv2

import sys
import os


def get_score(model_path, image_bytes):
    np_array = np.frombuffer(image_bytes, np.uint8)
    im0 = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    model = onnx.load(model_path)
    sess = ort.InferenceSession(model_path)
    h, w = im0.shape[:2]
    len_resize = 256
    len_crop = 224
    # bgr->rgb
    im0 = im0[:, :, ::-1]
    im0 = cv2.resize(im0, (len_resize, len_resize))
    # center crop
    center_y = int(len_resize / 2)
    center_x = int(len_resize / 2)
    tl_y = max(0, int(center_y - len_crop / 2))
    tl_x = max(0, int(center_x - len_crop / 2))
    im0 = im0[tl_y:(tl_y + len_crop), tl_x:(tl_x + len_crop), :]

    # h,w,c --> c, h, w
    im0 = np.transpose(im0, (2, 0, 1))
    # add batch dim
    im0 = np.expand_dims(im0.astype('float32'), axis=0)
    h = np.expand_dims(np.array([h]).astype('int32'), axis=0)
    w = np.expand_dims(np.array([w]).astype('int32'), axis=0)

    # [2/2] forward onnx
    score = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: im0, \
                                                    sess.get_inputs()[1].name: h, sess.get_inputs()[2].name: w})
    return score[0][0][0]


def inference_onnx(image_bytes):
    base_path = os.getcwd() + "\\Python\\inference_full\\"
    model_path_prismy_v3_nondefect = base_path + 'PrismyV3NonDefect_full.onnx'
    score_prismy_v3_nondefect = get_score(model_path_prismy_v3_nondefect, image_bytes)
    model_path_prismy_v3_rank = base_path + 'PrismyV3Rank_full.onnx'
    score_prismy_v3_rank = get_score(model_path_prismy_v3_rank, image_bytes)
    print(f'{score_prismy_v3_rank:.2f}')
    print(f'{score_prismy_v3_nondefect:.2f}')


if __name__ == '__main__':
    image_bytes = sys.stdin.buffer.read()
    inference_onnx(image_bytes)
