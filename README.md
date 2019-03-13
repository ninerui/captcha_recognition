# captcha_recognition
深度学习运用于验证码识别

[train_net.py](/train_net.py)----------训练网络的脚本  
[test_net.py](/test_net.py)------------一张一张的进行模型测试  
[data_helpers](/data_helpers.py)-------数据处理, 获取每一个batch数据   
[set_params](/set_params.py)-----------参数设置的地方

原始图片大小采用了(60, 160, 3)大小的三通道rgb图, 由于图片的颜色对于验证码的识别没有帮助,
所以将rgb图片转为单通道的灰度图


