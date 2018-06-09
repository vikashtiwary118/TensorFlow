# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
sess=tf.InteractiveSession()

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)



import matplotlib.pyplot as plt

def display_sample(num):
    print(mnist.train.labels[num])
    label=mnist.train.labels[num].argmax(axis=0)
    image=mnist.train.images[num].reshape([28,28])
    plt.title('Sample: %d Label : %d'%(num,label))
    plt.imshow(image,cmap=plt.get_cmap('gray_r'))
    plt.show()
display_sample(1234)

import numpy as np

images=mnist.train.images[0].reshape([1,784])
for i in range(1,500):
    images=np.concatenate((images,mnist.train.images[i].reshape([1,784])))
plt.imshow(images,cmap=plt.get_cmap('gray_r'))
plt.show()
