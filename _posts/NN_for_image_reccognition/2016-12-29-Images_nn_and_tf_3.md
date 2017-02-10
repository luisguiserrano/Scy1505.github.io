---
layout: post
title:  "NN for image reccognition - Logistic Regression"
date:   2016-12-28
category: NN_for_image_reccognition
---

This is part 3 of our blogpost related with images and tensorflow. The posts follow the following:

1. [Getting the Data]({{ site.baseurl }}{% post_url 2016-12-15-Images_nn_and_tf_1 %}).
2. [k-neareast Neighbor]({{ site.baseurl }}{% post_url 2016-12-22-Images_nn_and_tf_2 %}).
3. **Logistic Regression**.
4. [A two layer Neural Network]({{ site.baseurl }}{% post_url 2017-01-09-Images_nn_and_tf_4 %}).
5. [Convolutions in Tensorflow]({{ site.baseurl }}{% post_url 2017-01-16-Images_nn_and_tf_5 %}).
6. [Convolutional Network]({{ site.baseurl }}{% post_url 2017-01-23-Images_nn_and_tf_6 %}).

# Logistic Regression

A good reference can be found in the notes [here](http://cs231n.github.io/linear-classify/). In this notebook we implement Logistic Regression using Tensorflow. In this **other post** we did a logistic classification model to solve an Iris Data, we follow the same ideas. We start by importing the necessary libraries.


{% highlight ruby %}
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import aux
{% endhighlight %}

And of course, our data, which for logistic regression we need it as a flat array,


{% highlight ruby %}
X_train,y_train,X_test,y_test =aux.input(flat=True)
{% endhighlight %}

We need to modify y_train. The reason for this, is that y_train is a vector whose i-th value corresponds to the number associated with the class we want. Instead, we want that each output is a 10-dimensional array with a 1 in the axis associated with the correct label and zero otherwise, we can transfor this easy as follows


{% highlight ruby %}
y_train=(np.arange(10)==y_train[:,None]).astype(np.float32)
y_test=(np.arange(10) ==np.array(y_test)[:,None]).astype(np.float32)
{% endhighlight %}

As we want to use TensorFlow, we need to create our model using tensors. Recall that a *tensor* is a type of multidimensional array.  

## The Placeholders

We create the placeholders that holds our data.


{% highlight ruby %}
X_train_tf=tf.placeholder(tf.float32, [None,3072] )
X_test_tf=tf.placeholder(tf.float32, [None,3072])
y_train_tf=tf.placeholder(tf.float32,[None,10])
y_test_tf=tf.placeholder(tf.float32,[None,10])

{% endhighlight %}

## The Variables

We also create the variables, that is the matrix of weights and the bias vector.


{% highlight ruby %}
W=tf.Variable(tf.truncated_normal([3072,10]))
b = tf.Variable(tf.zeros([10]))
{% endhighlight %}

### The activation function

The logistic model uses the softmax activation function.


{% highlight ruby %}
y = tf.nn.softmax(tf.matmul(X_train_tf, W) + b)
{% endhighlight %}

### The cost function

For the cost function we use the log cross entropy.


{% highlight ruby %}
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_train_tf))
{% endhighlight %}

### Optimizer

We have many options here, for now we use the gradient descent with constant learning rate.


{% highlight ruby %}
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
{% endhighlight %}

### Checking on the test cases

We can compute the tensor of probabilities for the test cases by making


{% highlight ruby %}
test_prediction = tf.nn.softmax(tf.matmul(X_test_tf, W) + b)
{% endhighlight %}

and its accuracy by


{% highlight ruby %}
def accuracy(A,B):
    return 100*np.sum(np.argmax(A,1)==np.argmax(B,1))/A.shape[0]
{% endhighlight %}

### The training

We start the session.


{% highlight ruby %}
sess=tf.Session()
{% endhighlight %}

Initialize all the variables


{% highlight ruby %}
sess.run(tf.global_variables_initializer())
{% endhighlight %}

And run the training for a number of epochs


{% highlight ruby %}
EPOCHS=1001

for i in range(EPOCHS):
    _,y_ = sess.run([optimizer,y],feed_dict={X_train_tf:X_train,y_train_tf:y_train})
    if i%100==0:
        print("The accuracy at step %d for the training set is: %2.f%%"%(i,accuracy(y_train,y_)))

{% endhighlight %}

    The accuracy at step 0 for the training set is:  9%
    The accuracy at step 100 for the training set is: 10%
    The accuracy at step 200 for the training set is: 11%
    The accuracy at step 300 for the training set is: 14%
    The accuracy at step 400 for the training set is: 16%
    The accuracy at step 500 for the training set is: 17%
    The accuracy at step 600 for the training set is: 17%
    The accuracy at step 700 for the training set is: 18%
    The accuracy at step 800 for the training set is: 18%
    The accuracy at step 900 for the training set is: 18%
    The accuracy at step 1000 for the training set is: 19%


Finally, we check how well we did with the testing data


{% highlight ruby %}
test_pred=sess.run(test_prediction,feed_dict={X_test_tf:X_test})
print("The accuracy for the testing set is: %2.f%%"%(accuracy(y_test,test_pred)))
{% endhighlight %}

    The accuracy for the testing set is: 19%



{% highlight ruby %}
sess.close()
{% endhighlight %}

## Conclusions:

Logistic regresion is easy to implement, and even though it increases the accuracy, it doesn't do it by much. The idea now is to build more complicated networks in order to perform better.

## What can we do better?

Since neural networks can learn any function a good idea would be to enlarge the network. But, that is not the only thing we can improve:
- Use different optimization to our learning process, for example we could have a decaying learning or momentum. 
- Preprocess the data.
- Use a different activation function.
- Use Convolutional Neural Networks
In the next post we apply all of these techniques but the last one. We study convolutional networks in its own post later.


{% highlight ruby %}

{% endhighlight %}
