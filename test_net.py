import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import set_params
import data_helpers
from model_nets import cnn_net

if __name__ == '__main__':
    args = set_params.TestNet()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    text, image = data_helpers.gen_captcha_text_and_image(args.chars, args.char_count)
    print("验证码图像channel:", image.shape)  # (60, 160, 3)
    image_height, image_width, image_channel = image.shape
    image_shape = (image_height, image_width, 1)
    # 展示图片
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)
    plt.show()
    image = data_helpers.convert2gray(image)
    image = np.reshape(image, image_shape) / 255
    output_node = args.char_count * len(args.chars)
    X = tf.placeholder(tf.float32, (None, image_height, image_width, 1), "inputs")
    Y = tf.placeholder(tf.float32, (None, output_node))
    output = cnn_net.crack_captcha_cnn(X, output_node, keep_prob=1.0)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./save_models/crack_capcha.model-1710")
        # 获取预测结果
        predict = tf.argmax(tf.reshape(output, [-1, 4, 10]), 2)
        text_list = sess.run(predict, feed_dict={X: [image]})
        predict_text = text_list[0].tolist()

    print("正确: {}  预测: {}".format(text, predict_text))
