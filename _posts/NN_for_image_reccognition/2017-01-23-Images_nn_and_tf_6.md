---
layout: post
title:  "NN for image reccognition - Part 6"
date:   2017-01-23
category: NN_for_image_reccognition
---

This is part 6 of our blogpost related with images and tensorflow. The posts follow the following:

1. Getting the Data.
2. k-neareast Neighbor.
3. Logistic Regression.
4. A two layer Neural Network.
5. Convolutions in Tensorflow.
6. **Convolutional Networks.**
7. What's next?

# Convolutional Networks.

Before dwelling into this post, I recommend that you read the module two of the excellent introductory course [CS231n](http://cs231n.github.io/).

In this, our last post of this series, we develop a convolutional network to attack the image classificiation of CIFAR10. There are many resources and results out there, some of them reaching high accuracy. In order to make things simple, we have a more naive approach. But we include some of the features we have seen before together with some new ones.

- We keep sizes small for practical purposes.
- We create wrappers to have a cleaner code.
- Preprocess of image.
- We use Tensorboard to create summaries of our results.
- We use Decay Rate or adamOptimizer?
- We include maxpooling.
- We include dropout.
- We include code for saving the trained weights.

## Preparations 

We load the necessary libraries 


{% highlight ruby %}
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import aux
{% endhighlight %}

and import the data


{% highlight ruby %}
X_train,y_train,X_test,y_test =aux.input(flat=False)
{% endhighlight %}

And, as before we modify our labels


{% highlight ruby %}
y_train=(np.arange(10)==y_train[:,None]).astype(np.float32)
y_test=(np.arange(10) ==np.array(y_test)[:,None]).astype(np.float32)
{% endhighlight %}

# Preprocess Image

There are many operations that can be done here to preprocess an image (image augmentation, blurring,etc.) We opt only for normalizing the images in each channel. That is we fix each channel, and each pixel, and compute  $$\frac{x-\text{E}[x]}{\sigma[x]+\epsilon}$$

We first compute the mean and standard deviation for each entry


{% highlight ruby %}
mean = np.mean(X_train,axis=0)
std = np.std(X_train,axis=0)
{% endhighlight %}

And then normalize the train and testing data


{% highlight ruby %}
X_train= (X_train-mean)/(std+1e-16)
X_test= (X_test-mean)/(std+1e-16)
{% endhighlight %}

# The Network

We are ready to begin creating our network. We need to decide on a style, so we go for 
**<p style="text-align: center;"> INPUT -> [CONV -> RELU -> MAXPOOL]*2 -> FULLY CONNECTED -> RELU -> FULLY CONNECTED </p>**

We need to create placeholders to hold the imput, then we will have two layers consisting of a convolutional network with rectified linear unit activations and a max pool. We follow by a fully connected network with rectified linear output and we finish with a fully coneceted netwkork with sigmoid activation. It is important that we keep track of the different sizes. 

| **Tensor** | **Size**|
|:----:|:---:|
|Input|32x32x3|
|After First Convolution | 32x32x16|
|After Second Convolution | 32x32x32| 

Double check, I don't think this is right, because of max pool!

# The global parameters


It is a good idea to keep the global parameters as constants. A question is what global parameters we need. 


{% highlight ruby %}
# Training Parameters
learning_rate=0.001
epochs=1001
display_step=1000


#Network Parameters
n_input=len(X_train)
n_classes=10
dropout=0.75

#The tensor keeping the dropout probability.
keep_prob = tf.placeholder(tf.float32)

#A path for saving the log files for TensorBoard
logs_path = './conv_logs/'
model_path = "./conv_model/model.ckpt"

{% endhighlight %}

## The Placeholders

We create the placeholders that holds our data.


{% highlight ruby %}
X_train_tf=tf.nn.l2_normalize(tf.placeholder(tf.float32, [None,32,32,3],name='X_train'),2)
X_test_tf=tf.placeholder(tf.float32, [None,32,32,3],name='X_test')
y_train_tf=tf.placeholder(tf.float32,[None,10],name='y_train')
y_test_tf=tf.placeholder(tf.float32,[None,10],name='y_test')
{% endhighlight %}

# The Convolution Matrices

We have two convolution layers, hence we need two convolution matrices


{% highlight ruby %}
weights={}

# 3x3 conv, 3 input, 16 outputs
weights['wc1']=tf.Variable(tf.random_normal([3, 3, 3, 16]),name='wc1')

# 5x5 conv, 16 inputs, 32 outputs
weights['wc2']=tf.Variable(tf.random_normal([5, 5, 16, 32]),name='wc2')



{% endhighlight %}

# The Later Layer Weights

We also need the weights for the matrix in the connected layer. Note that the two max pooling reduce the size of the tensor from 32x32 to 8x8.


{% highlight ruby %}
# fully connected, 8*8*32 inputs, 1024 outputs
weights['wd1'] = tf.Variable(tf.random_normal([8*8*32, 256]),name='wd1')

# # 1024 inputs, 10 outputs (class prediction)
weights['out'] = tf.Variable(tf.random_normal([256, n_classes]),name='wout')

{% endhighlight %}

# The Biases Vectors

Each layer needs a bias vector


{% highlight ruby %}
biases = {
    'bc1': tf.Variable(tf.random_normal([16]),name='bc1'),
    'bc2': tf.Variable(tf.random_normal([32]),name='bc2'),
    'bd1': tf.Variable(tf.random_normal([256]),name='bd1'),
    'out': tf.Variable(tf.random_normal([n_classes]),name='bout')
}

{% endhighlight %}

# The Network wrappers

We want our network to be easy to read, understand, and debug, we use wrappers to acchieve this. In particular, we wrap the layers, which each consists of a convulational part and a relu activation function, and in some cases a maxpool.


{% highlight ruby %}
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)



def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


{% endhighlight %}

# The Network code

We are ready to have our main code.


{% highlight ruby %}
def conv_net(x, weights, biases, dropout):

    # Convolution Layer 1
    with tf.name_scope('Convolution_1'):
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

    with tf.name_scope('Convolution_2'):    
        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)
    
    with tf.name_scope('Fully_Connected'):
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        
        # Apply Dropout
        with tf.name_scope('Dropout'):
            fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

with tf.name_scope('Model'):
    # Construct model
    pred = conv_net(X_train_tf, weights, biases, keep_prob)

{% endhighlight %}

# Loss function, Optimizer, and Accuracy

We use cross entropy with logits for the lost function and the Adam Optimizer for updating the weights. 


{% highlight ruby %}
with tf.name_scope('Loss'):
    #Define the cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_train_tf))

with tf.name_scope('Optimizer'):
    #Defines the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_train_tf, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


{% endhighlight %}

# Evaluating the model

We initialize the variables


{% highlight ruby %}
# Initializing the variables
init = tf.global_variables_initializer()
{% endhighlight %}

And create some summaries and the op for putting them together.


{% highlight ruby %}
#We keep track of the cost and accuracy
tf.summary.scalar("loss", cost)
tf.summary.scalar("accuracy", accuracy)


# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
    

# The op for merging the summaries
merged_summary_op = tf.summary.merge_all()

{% endhighlight %}

    INFO:tensorflow:Summary name wc1:0 is illegal; using wc1_0 instead.
    INFO:tensorflow:Summary name wc2:0 is illegal; using wc2_0 instead.
    INFO:tensorflow:Summary name wd1:0 is illegal; using wd1_0 instead.
    INFO:tensorflow:Summary name wout:0 is illegal; using wout_0 instead.
    INFO:tensorflow:Summary name bc1:0 is illegal; using bc1_0 instead.
    INFO:tensorflow:Summary name bc2:0 is illegal; using bc2_0 instead.
    INFO:tensorflow:Summary name bd1:0 is illegal; using bd1_0 instead.
    INFO:tensorflow:Summary name bout:0 is illegal; using bout_0 instead.


As we want to reuse the values we get in the variables, we need an operation for saving.


{% highlight ruby %}
# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()
{% endhighlight %}

Finally, we run our model, we will just use a small set of data. (If you have a GPU you can run the model on the whole dataset for a couple of weeks to get a nice result)


{% highlight ruby %}
# We get only one fifth of the data.
n_input=1500

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
     # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path,
                                            graph=tf.get_default_graph())
    
    for i in range(epochs):
        print('\r This is epoch %d'%(i), end='. ')
        indexes=np.random.randint(0,n_input,500)
        sess.run(optimizer,feed_dict={X_train_tf:X_train[:n_input][indexes],y_train_tf:y_train[:n_input][indexes],keep_prob: dropout})
    

        if i%display_step ==0:
            # Calculate batch loss and accuracy
            loss, acc,summary = sess.run([cost, accuracy,merged_summary_op], feed_dict={X_train_tf: X_train[:n_input][indexes],
                                                              y_train_tf: y_train[:n_input][indexes],
                                                              keep_prob: 1.0 })
            summary_writer.add_summary(summary, i)
            print("Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
            
    
   
{% endhighlight %}

     This is epoch 0. Minibatch Loss= 38566.578125, Training Accuracy= 0.08600
     This is epoch 1000. Minibatch Loss= 3.023288, Training Accuracy= 0.28600
    Model saved in file: ./conv_model/model.ckpt


As we saved our model, we could reuse it and run many more epochs. For example we could (but don't) use the following.


{% highlight ruby %}
epochs=2001
display_step=200
{% endhighlight %}


{% highlight ruby %}
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    
    # Restore model weights from previously saved model
    load_path = saver.restore(sess, model_path)
    print("Model restored from file: %s" % save_path)
    
    
     # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path,
                                            graph=tf.get_default_graph())
    
    
    for i in range(epochs):
        print('\r This is epoch %d'%(i+1001), end='. ')
        indexes=np.random.randint(0,n_input,500)
        sess.run(optimizer,feed_dict={X_train_tf:X_train[:n_input][indexes],y_train_tf:y_train[:n_input][indexes],keep_prob: dropout})
        

        if i%display_step ==0:
            # Calculate batch loss and accuracy
            loss, acc,summary = sess.run([cost, accuracy,merged_summary_op], feed_dict={X_train_tf: X_train[:n_input][indexes],
                                                              y_train_tf: y_train[:n_input][indexes],
                                                              keep_prob: 1.})
            summary_writer.add_summary(summary, i)
            print("Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
              
{% endhighlight %}

    Model restored from file: ./conv_model/model.ckpt
     This is epoch 1001. Minibatch Loss= 2.520611, Training Accuracy= 0.31200
     This is epoch 1201. Minibatch Loss= 1.945457, Training Accuracy= 0.29000
     This is epoch 1401. Minibatch Loss= 1.890653, Training Accuracy= 0.28400
     This is epoch 1601. Minibatch Loss= 1.898642, Training Accuracy= 0.26600
     This is epoch 1801. Minibatch Loss= 1.908458, Training Accuracy= 0.25400
     This is epoch 2001. Minibatch Loss= 1.848357, Training Accuracy= 0.28200
     This is epoch 2201. Minibatch Loss= 1.952290, Training Accuracy= 0.25000
     This is epoch 2401. Minibatch Loss= 1.889921, Training Accuracy= 0.26000
     This is epoch 2601. Minibatch Loss= 1.880741, Training Accuracy= 0.28200
     This is epoch 2801. Minibatch Loss= 1.920311, Training Accuracy= 0.24600
     This is epoch 3001. Minibatch Loss= 1.870832, Training Accuracy= 0.27800
    Model saved in file: ./conv_model/model.ckpt


We finish by using our trained model to predict the results in the testing data


{% highlight ruby %}
with tf.Session() as sess:
    sess.run(init)
    
    
    # Restore model weights from previously saved model
    load_path = saver.restore(sess, model_path)
    print("Model restored from file: %s" % save_path)
    
    pred_test = conv_net(X_test_tf, weights, biases, keep_prob)
    correct_pred_test = tf.equal(tf.argmax(pred_test, 1), tf.argmax(y_test_tf, 1))
    accuracy_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32))
    
    acc=sess.run(accuracy_test,feed_dict={X_test_tf: X_test,y_test_tf: y_test,keep_prob: 1.})
    
    print('The accuracy for the test set is %.3f'%acc)
    
{% endhighlight %}

    Model restored from file: ./conv_model/model.ckpt
    The accuracy for the test set is 0.149


# What's next?

You should learn to use GPU. Explore imagenet, use pretrained models and adapt them to yours, learn Keras. Study NPL problems, Recurrent Networks, etc. There are many things to learn, and many examples online. 

Finally, you should check out [this awesome lecture](https://www.youtube.com/watch?v=u6aEYuemt0M).
