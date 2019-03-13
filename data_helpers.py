import random

import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha


def random_captcha_text(char_set, captcha_size=4):
    """
    随机选取数字或者字母
    :param char_set:
    :param captcha_size:
    :return:
    """
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image(char_set, captcha_size=4):
    image = ImageCaptcha()

    captcha_text = random_captcha_text(char_set, captcha_size=captcha_size)
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text + '.jpg')

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


def convert2gray(img):
    if len(img.shape) > 2:
        # gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def text2vec(text, output_node):
    # text_len = len(text)
    # if text_len > MAX_CAPTCHA:
    #     raise ValueError('验证码最长4个字符')

    vector = np.zeros(output_node)
    """
    def char2pos(c):  
        if c =='_':  
            k = 62  
            return k  
        k = ord(c)-48  
        if k > 9:  
            k = ord(c) - 55  
            if k > 35:  
                k = ord(c) - 61  
                if k > 61:  
                    raise ValueError('No Map')   
        return k  
    """
    for i, c in enumerate(text):
        idx = i * 10 + int(c)
        vector[idx] = 1
    return vector


# 生成一个训练batch
def get_next_batch(char_set, captcha_size, image_shape, label_shape, batch_size=128):
    batch_x = np.zeros([batch_size, image_shape[0], image_shape[1], image_shape[2]])
    batch_y = np.zeros([batch_size, label_shape[0]])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image(char_set, captcha_size=captcha_size)
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i, :] = np.reshape(image, image_shape) / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text, label_shape[0])

    return batch_x, batch_y
