#coding:utf-8
from tensorflow.examples.tutorials.mnist import input_data

#获取mnist数据
mnist =input_data.read_data_sets("./MNIST_DATA/", one_hot = True)

#查看训练数据大小
print(mnist.train.images.shape) #55000, 784
print(mnist.train.labels.shape) #55000, 10

#查看验证数据大小
print(mnist.validation.images.shape) #5000, 784
print(mnist.validation.labels.shape) #5000, 10

#查看测试集大小
print(mnist.test.images.shape) #10000, 784
print(mnist.test.labels.shape) #10000, 10
