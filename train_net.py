import tensorflow as tf
# import matplotlib.pyplot as plt

import set_params
import data_helpers
from model_nets import cnn_net

if __name__ == '__main__':
    args = set_params.TrainNet()

    text, image = data_helpers.gen_captcha_text_and_image(args.chars, args.char_count)
    print("验证码图像channel:", image.shape)  # (60, 160, 3)
    image_height, image_width, image_channel = image.shape
    # plt.imshow(image)
    # plt.show()
    output_node = args.char_count * len(args.chars)
    X = tf.placeholder(tf.float32, (None, image_height, image_width, 1), "inputs")
    Y = tf.placeholder(tf.float32, (None, output_node))
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    output = cnn_net.crack_captcha_cnn(X, output_node, keep_prob=keep_prob)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    predict = tf.reshape(output, [-1, args.char_count, len(args.chars)])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, args.char_count, len(args.chars)]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            batch_x, batch_y = data_helpers.get_next_batch(
                args.chars, args.char_count, (image_height, image_width, 1), (output_node,), batch_size=args.batch_size)
            _, loss_ = sess.run([optimizer, loss],
                                feed_dict={X: batch_x, Y: batch_y, keep_prob: args.keep_prob, learning_rate: 0.001})
            print(step, loss_)

            # 每100 step计算一次准确率
            if step % 10 == 0:
                batch_x_test, batch_y_test = data_helpers.get_next_batch(
                    args.chars, args.char_count, (image_height, image_width, 1), (output_node,),
                    batch_size=args.test_batch_size)
                acc = sess.run(accuracy,
                               feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1., learning_rate: 0.001})
                print(step, acc)
                # 如果准确率大于90%,保存模型,完成训练
                if acc > 0.90:
                    saver.save(sess, "./save_model/crack_capcha.model", global_step=step)
                    break
            step += 1
