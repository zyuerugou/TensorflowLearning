#coding:utf-8
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#获取mnist数据，如果不存在会先下载
mnist =input_data.read_data_sets("./MNIST_DATA/", one_hot = True)



#获取前20张图片的label
for i in range(20):
    #得到独热表示，形如(0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    OneHotLabel = mnist.train.labels[i, :]
    #通过np.argmax，可以直接获得原始的label
    #因为只有1位为1，其他的都是0
    label = np.argmax(OneHotLabel)
    print('mnist_train_%d.jpg label: %d' % (i, label))
