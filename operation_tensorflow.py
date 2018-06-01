# -*- coding: utf-8 -*-
import tensorflow as tf
Input=tf.placeholder('float',shape=[None,2],name='Input')
target=tf.placeholder('float',shape=[None,1],name='target')

input_bias=tf.Variable(initial_value=tf.random_normal(shape=[3],stddev=0.4),dtype='float',name='input_bias')
weights=tf.Variable(initial_value=tf.random_normal(shape=[2,3],stddev=0.4),dtype='float',name='hidden_weight')
hidden_bias=tf.Variable(initial_value=tf.random_normal(shape=[1],stddev=0.4),dtype='float',name='hidden_bias')

outputWeights=tf.Variable(initial_value=tf.random_normal(shape=[3,1],stddev=0.4),dtype='float',name='output_weights')

hiddenLayer=tf.matmul(Input,weights)+input_bias
hiddenLayer=tf.sigmoid(hiddenLayer,name='hidden_layer_activation')

output=tf.matmul(hiddenLayer,outputWeights)+hidden_bias
output=tf.sigmoid(output,name='outpu_layer_activation')

cost=tf.squared_difference(target,output)
cost=tf.reduce_mean(cost)
optimizer=tf.train.AdamOptimizer().minimize(cost)


inp=[[1,1],[1,0],[0,1],[0,0]]
out=[[0],[1],[1],[0]] #for XOR operation
epochs=4000
# we have to create the session to run the network
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epochs):
            error,_=sess.run([cost,optimizer], feed_dict={Input:inp,target:out})
            print(i,error)
        
    while True:
        inp=[[0,0]]
        inp[0][0]=input('Type 1st input')
        inp[0][1]=input('Type 2st input')
        print(sess.run([output], feed_dict={Input:inp})[0][0])
        

        
