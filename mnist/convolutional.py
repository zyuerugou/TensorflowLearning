# coding: utf-8

#导入Tensorflow
#这句是导入Tensorflow约定俗成的做法
import tensorflow as tf

#导入MNIST教学模块
from tensorflow.examples.tutorials.mnist import input_data
#读入MNIST数据
mnist =input_data.read_data_sets("./MNIST_DATA/", one_hot = True)


#--------------------------------------------------
#创建模型
#--------------------------------------------------
#创建x，x是一个占位符（placeholder），代表待识别的图片
x = tf.placeholder(tf.float32, [None, 784])
#y_是实际的图像标签，同样以占位符表示
y_ = tf.placeholder(tf.float32, [None, 10])

#将单张图片从784维向量转换为28*28的矩阵图片
x_image = tf.reshape(x, [-1, 28, 28, 1])



# function:	生成一个给定形状的卷积核变量，并自动以截断正态分布初始化
# return:	卷积核
# parameter:	
#	shape:	给定的形状
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# function:	生成一个给定形状的偏置变量，并初始化所有值为0.1
# return:	偏置
# parameter:	
#	shape:	给定的形状
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# function:	计算卷积
# return:	卷积结果
# parameter:	
#	x:	输入
#	W:	卷积核
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

# function:	池化
# return:	池化结果
# parameter:	
#	x:	卷积结果
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


#第一层卷积层
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
#卷积后选用relu函数作为激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#池化
h_pool1 = max_pool_2x2(h_conv1)


#第二层卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#全连接层，输出为1024维的向量
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#使用dropout。keep_prob是一个占位符，训练时为0.5，测试时为1.0
#dropout是用来防止神经网络过拟合的，在每一步训练时，以一定概率去掉网络中的某些连接（在当前步骤中随机去除）
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#第二层全连接，将1024维的向量h_fc1_drop转换为10维的得分向量
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


#不采用先softmax再计算交叉熵的方法
#而是用tf.nn.softmax_cross_entropy_with_logits直接计算
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))
#同样定义train_step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#定义测试的准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#----------------------------------------------------------------------
#训练
#----------------------------------------------------------------------
#创建Session,对变量初始化
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#训练20000步
for i in range(20000):
    batch = mnist.train.next_batch(50)
    #每100步报告一次在验证集上的准确率
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step: %d, training accuracy: %g' % (i, train_accuracy))
    train_step.run(feed_dict = {x : batch[0], y_ : batch[1], keep_prob: 0.5})


#----------------------------------------------------------------------
#测试
#----------------------------------------------------------------------
print('test accuracy: %g' % accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))




