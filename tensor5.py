# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
sess=tf.InteractiveSession()

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

input_images=tf.placeholder(tf.float32,shape=[None,784])
target_labels=tf.placeholder(tf.float32,shape=[None,10])

hidden_nodes=512
input_weight=tf.Variable(tf.truncated_normal([784,hidden_nodes]))
input_biases=tf.Variable(tf.zeros(hidden_nodes))

hidden_weight=tf.Variable(tf.truncated_normal([hidden_nodes,10]))
hidden_biases=tf.Variable(tf.zeros(10))

input_layer=tf.matmul(input_images,input_weight)
hidden_layer=tf.nn.relu(input_layer+input_biases)
digit_weights=tf.matmul(hidden_layer,hidden_weight)+hidden_biases


loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=digit_weights, labels=target_labels))

optimizer=tf.train.GradientDescentOptimizer(0.5).minimize(loss_function)

correct_prediction=tf.equal(tf.arg_max(digit_weights,1),tf.arg_max(target_labels,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.global_variables_initializer().run()

for x in range(2000):
    batch=mnist.train.next_batch(100)
    optimizer.run(feed_dict={input_images: batch[0],target_labels: batch[1]})
    if((x+1)%100==0):
        print('Training epoch'+str(x+1))
        print('Accuracy'+str(accuracy.eval(feed_dict={input_images: mnist.test.images,target_labels:mnist.test.labels})))


for x in range(100):
    x_train=mnist.test.images[x,:].reshape(1,784) 
    y_train=mnist.test.labels[x,:]
    label=y_train.argmax()
    prediction=sess.run(digit_weights,feed_dict={input_images:x_train}).argmax()
    if(prediction!=label):
        plt.title('Prediction : %d Label : %d'%(prediction,label))
        plt.imshow(x_train.reshape([28,28]),cmap=plt.get_cmap('gray_r'))
        plt.show()