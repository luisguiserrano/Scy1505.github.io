---
layout: post
title:  "NN for image reccognition - Part 4"
date:   2017-1-09
category: convolutional_networks
---

This is part 4 of our blogpost related with images and tensorflow. The posts follow the following:

1. Getting the Data.
2. k-neareast Neighbor.
3. Logistic Regression.
4. **A two layer Neural Networks.**
5. Convolutions in Tensorflow.
6. Convolutional Networks.
7. What's next?

# A two layer Neural Network.

In this post we built a two layer Neural Network to predict CIFAR10. A couple of remarks before we start:
- We keep the sizes small since training takes a long time.
- We use Tensorboard to keep track of all the info. See this other post for an intro to Tensorboard.
- We use a decaying learning rate. See this other post for an intro to learning rates in tensor flow.

## The design.

How many neurons should we choose for the first layer?, how many for the second? At this moment we don't have any reasons to choose any particular number over another

## Preparations

As always, we import the required libraries


{% highlight ruby %}
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import aux
{% endhighlight %}

and the data, that we need it to be flat as well for this example,


{% highlight ruby %}
X_train,y_train,X_test,y_test =aux.input(flat=True)
{% endhighlight %}

And as in the last post we transform our outputs


{% highlight ruby %}
y_train=(np.arange(10)==y_train[:,None]).astype(np.float32)
y_test=(np.arange(10) ==np.array(y_test)[:,None]).astype(np.float32)
{% endhighlight %}

## The Placeholders

We create the placeholders that holds our data.


{% highlight ruby %}
X_train_tf=tf.placeholder(tf.float32, [None,3072],name='X_train')
X_test_tf=tf.placeholder(tf.float32, [None,3072],name='X_test')
y_train_tf=tf.placeholder(tf.float32,[None,10],name='y_train')
y_test_tf=tf.placeholder(tf.float32,[None,10],name='y_test')
{% endhighlight %}

## The Variables

We also create the variables, that is the matrix of weights and the bias vectors. For convenience, we write W_ij to represent the matrix i on the layer j. Similarly b_ij represents the bias vector i on the layer j.


{% highlight ruby %}
W_11=tf.Variable(tf.random_normal([3072,128], mean=0, stddev=0.01), name='W_11')
W_21=tf.Variable(tf.random_normal([3072,128], mean=0, stddev=0.01), name='W_21')
W_31=tf.Variable(tf.random_normal([3072,128], mean=0, stddev=0.01), name='W_31')

W_12=tf.Variable(tf.random_normal([128,10], mean=0, stddev=0.01), name='W_12')


b_11 = tf.Variable(tf.random_normal([128], mean=0, stddev=0.01), name='b_11')
b_21 = tf.Variable(tf.random_normal([128], mean=0, stddev=0.01), name='b_21')
b_31 = tf.Variable(tf.random_normal([128], mean=0, stddev=0.01), name='b_31')

b_12 = tf.Variable(tf.random_normal([10], mean=0, stddev=0.01), name='b_12')
{% endhighlight %}

We create a summary of our variables


{% highlight ruby %}
tf.summary.histogram("W_11_Summary", W_11)
tf.summary.histogram("W_21_Summary", W_21)
tf.summary.histogram("W_31_Summary", W_31)

tf.summary.histogram("W_12_Summary", W_12)

tf.summary.histogram("b_11_Summary", b_11)
tf.summary.histogram("b_21_Summary", b_21)
tf.summary.histogram("b_31_Summary", b_31)

tf.summary.histogram("b_12_Summary", b_12);
{% endhighlight %}

## The model.

We create our model as a function and use it later during the training.


{% highlight ruby %}
def create_model(X, W_11,W_21,W_31,W_12,b_11,b_21,b_31,b_12):
    # Add layer name scopes for better graph visualization
    with tf.name_scope("layer_1"):
        with tf.name_scope("Neuron_11"):
            apply_weights_11 = tf.matmul(X, W_11, name="Apply_W_11")
            add_bias_11 = tf.add(apply_weights_11, b_11, name="add_b_11")
        
        with tf.name_scope("Neuron_21"):
            apply_weights_21 = tf.matmul(X, W_21, name="Apply_W_21")
            add_bias_21 = tf.add(apply_weights_21, b_11, name="add_b_21")
            
        with tf.name_scope("Neuron_31"):
            apply_weights_31 = tf.matmul(X, W_31, name="Apply_W_31")
            add_bias_31 = tf.add(apply_weights_31, b_31, name="add_b_31")

        with tf.name_scope("Activation_1"):
            activation_11 = tf.nn.relu(add_bias_11, name="activation_11")
            activation_21 = tf.nn.relu(add_bias_11, name="activation_21")
            activation_31 = tf.nn.relu(add_bias_11, name="activation_31")

            activation_1=activation_11+activation_21+activation_31

    with tf.name_scope("layer_2"):
        with tf.name_scope("Neuron_12"):
            apply_weights_12 = tf.matmul(activation_1, W_12, name="Apply_W_12")
            add_bias_12 = tf.add(apply_weights_12, b_12, name="add_b_12")
        
        with tf.name_scope("Activation_2"):
            activation_12 = tf.nn.sigmoid(add_bias_12, name="activation_12")
        
        
    return activation_12
{% endhighlight %}

## The cost function

We use cross entropy as our cost function.


{% highlight ruby %}
def cost_fun(y,y_true):
    with tf.name_scope("cost_function"):  
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y,
                                                        labels=y_true)

        cost = tf.reduce_mean(cross_entropy)
    return cost

{% endhighlight %}

## The accuracy function

A simple helper function to compute accuracy.


{% highlight ruby %}
def accuracy(A,B):
    return 100*np.sum(np.argmax(A,1)==np.argmax(B,1))/A.shape[0]
{% endhighlight %}

## The training

We now train the model, note the use of a learning rate with exponential decay.


{% highlight ruby %}
EPOCHS=1001

with tf.Session() as sess:
    
     #Creates the writter for the summaries
    writer = tf.summary.FileWriter("./logs", sess.graph) # 
    
    #puts all the summaries together
    merged = tf.summary.merge_all()

    #Initialize the variables
    tf.global_variables_initializer().run()
    
    #The model
    y=create_model(X_train_tf, W_11,W_21,W_31,W_12,b_11,b_21,b_31,b_12)
    
    #The cost
    cost=cost_fun(y,y_train_tf)
    
    
      
    for i in range(EPOCHS):
        
        #Creates the learning rate with decay
        learning_rate=tf.train.exponential_decay(learning_rate=0.1,
                                          global_step= i,
                                          decay_steps=32,
                                          decay_rate= 0.95,
                                          staircase=True)
        
        #Creates the training opt using gradient descent
        training = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
        
        #runs the training 
        sess.run(training,feed_dict={X_train_tf:X_train,y_train_tf:y_train})
        y_=sess.run(y,feed_dict={X_train_tf:X_train})
        
        if i%100==0:
            print("The accuracy at step %d for the training set is: %2.f%%"%(i,accuracy(y_train,y_)))
    
    test_prediction=create_model(X_test_tf, W_11,W_21,W_31,W_12,b_11,b_21,b_31,b_12)
    test_pred=sess.run(test_prediction,feed_dict={X_test_tf:X_test})
    print("The accuracy for the testing set is: %2.f%%"%(accuracy(y_test,test_pred)))
        
{% endhighlight %}

    The accuracy at step 0 for the training set is: 11%
    The accuracy at step 100 for the training set is: 26%
    The accuracy at step 200 for the training set is: 29%
    The accuracy at step 300 for the training set is: 30%
    The accuracy at step 400 for the training set is: 31%
    The accuracy at step 500 for the training set is: 31%
    The accuracy at step 600 for the training set is: 32%
    The accuracy at step 700 for the training set is: 33%
    The accuracy at step 800 for the training set is: 33%
    The accuracy at step 900 for the training set is: 34%
    The accuracy at step 1000 for the training set is: 34%
    The accuracy for the testing set is: 33%


As we see, we have improved over the logistic model we had before, but we choose the network blindly! We should study the problem better and come up with a network that exploits the uniqueness of the problem. We talk about this in the next blog. 

Ps: Check the tensorboard!
