# -*- coding: utf-8 -*-
import tensorflow as tf
Input_1=tf.placeholder('float',shape=[None,3],name='Input_1')
Input_2=tf.placeholder('float',shape=[None,3],name='Input_2')
x=tf.Variable(0,dtype='float')
output=tf.Variable(0,dtype='float')
x=Input_1*Input_2
output=x*x
ses=tf.Session()
print(ses.run(output,feed_dict={Input_1:[[1,2,3]],Input_2:[[4,5,6]]}))


