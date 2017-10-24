---
layout: post
title:  "NN for image reccognition - Convolutions in Tensorflow"
date:   2017-01-16
category: NN_for_image_reccognition
---

This is part 5 of our blogpost related with images and tensorflow. The posts follow the following:

1. [Getting the Data]({{ site.baseurl }}{% post_url 2016-12-15-Images_nn_and_tf_1 %}).
2. [k-neareast Neighbor]({{ site.baseurl }}{% post_url 2016-12-22-Images_nn_and_tf_2 %}).
3. [Logistic Regression]({{ site.baseurl }}{% post_url 2016-12-29-Images_nn_and_tf_3 %}).
4. [A two layer Neural Network]({{ site.baseurl }}{% post_url 2017-01-09-Images_nn_and_tf_4 %}).
5. **Convolutions in Tensorflow**.
6. [Convolutional Network]({{ site.baseurl }}{% post_url 2017-01-23-Images_nn_and_tf_6 %}).



# Convolutions in Tensorflow

Before dwelling into this post, I recommend that you read the module two of the excellent introductory course [CS231n](http://cs231n.github.io/).

We show how to use convolutions in Tensorflow by example.

## Images

Just as a reminder, images come in as a paralelepid array, some of the sort *Width x Height x Chanels*. The chanels, for the RGB encoding the chanels are red, green, and blue. Which intensity shows as a float between 0 and 1. The following gif makes an excellent point in showing how this works.


<center>
<img src="{{ '/assets/img/Images_nn_and_tf_5_files/RGB.gif' | prepend: site.baseurl }}" alt=""> 
</center>



## Convolutions

The goal is to find the right convolutions to be applied to every 3x3x1 section of the images. Let's see how tensorflow deals with convolutions, we use an example for this.


{% highlight ruby %}
from scipy import misc
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
{% endhighlight %}

Let's read an image, we can use the method [imread()](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.misc.imread.html) from the misc library inside scipy.


{% highlight ruby %}
im=misc.imread("baobei.JPG")
plt.imshow(im)
plt.xlabel(" Image ")
plt.show();
{% endhighlight %}


<center>
<img src="{{ '/assets/img/Images_nn_and_tf_5_files/Images_nn_and_tf_5_10_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


Let's see what the shape of the image is:


{% highlight ruby %}
im.shape
{% endhighlight %}




    (3265, 2177, 3)



To make an image a tensor, we just treat it as an array. So we need to have a placeholder.


{% highlight ruby %}
im_tf=tf.placeholder(np.float32,im.shape)
{% endhighlight %}

We first make the image channel be floats between zero and one. We can acchive this by making:


{% highlight ruby %}
im_float=im.astype(np.float)/255.0
{% endhighlight %}

But as we saw above, this image is too large, we can resize it by using [resize_images()](https://www.tensorflow.org/api_docs/python/image/resizing#resize_images).


{% highlight ruby %}
with tf.Session() as sess:
    im_resized=sess.run(tf.image.resize_images(im_tf,[408,272]), feed_dict={im_tf:im_float})

plt.imshow(im_resized)
plt.xlabel(" Image ")
plt.show();

{% endhighlight %}


<center>
<img src="{{ '/assets/img/Images_nn_and_tf_5_files/Images_nn_and_tf_5_18_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


Now we can play with different convolutions. We use the following 3x3 ones. 

$$ C_1=\begin{pmatrix} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0\end{pmatrix},\; C_2=\frac{1}{9}\begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1\end{pmatrix},\; C_3=\begin{pmatrix} -1/9 & -1/9 & -1/9 \\ -1/9 & 2-1/9 & -1/9 \\ -1/9 & -1/9 & -1/9\end{pmatrix},\; C_4=\begin{pmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1\end{pmatrix} $$

Or in code we can write this as 


{% highlight ruby %}
C_1=(np.array([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])).astype(np.float32)
C_2=np.array([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]]).astype(np.float32)/9
C_3=np.array([[-1.0/9,-1.0/9,-1.0/9],[-1.0/9,2-1.0/9,-1.0/9],[-1.0/9,-1.0/9,-1.0/9]]).astype(np.float32)
C_4=np.array([[-1.0,-2.0,-1.0],[0.0,0.0,0.0],[1.0,2.0,1.0]]).astype(np.float32)
{% endhighlight %}

We want to use [depthwise_conv2d()](https://www.tensorflow.org/api_docs/python/nn/convolution#depthwise_conv2d), to do so we want to apply the filter C_i to each of the channels we have, and obtain a channel of the same form, to do so we need to create a 3x3x3x1 array, that is constant along the last two axis. We can do this as follows:


{% highlight ruby %}
def creating(C):
    A=np.array([[C],[C],[C]])
    A=A.transpose([2,3,0,1])
    return A
{% endhighlight %}

We make them tensors and add the image.


{% highlight ruby %}
C_1_tf_0=tf.Variable(creating(C_1),dtype=tf.float32)
C_2_tf_0=tf.Variable(creating(C_2),dtype=tf.float32)
C_3_tf_0=tf.Variable(creating(C_3),dtype=tf.float32)
C_4_tf_0=tf.Variable(creating(C_4),dtype=tf.float32)
im_tf_0=tf.Variable(im_resized,dtype=tf.float32)
{% endhighlight %}

Nnote that we need to make the tensor to be 4d to apply this. We achive the last part via [expand_dims()](https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#expand_dims)


{% highlight ruby %}
im_tf=tf.expand_dims(im_tf_0,0)
{% endhighlight %}

We can now apply the convolutions


{% highlight ruby %}

with tf.Session() as sess:
    ConOut_1=tf.nn.depthwise_conv2d(input=im_tf,filter=tf.to_float(C_1_tf_0),strides=[1, 1, 1, 1], padding='SAME')
    ConOut_2=tf.nn.depthwise_conv2d(input=im_tf,filter=tf.to_float(C_2_tf_0),strides=[1, 1, 1, 1], padding='SAME')
    ConOut_3=tf.nn.depthwise_conv2d(input=im_tf,filter=C_3_tf_0,strides=[1, 1, 1, 1], padding='SAME')
    ConOut_4=tf.nn.depthwise_conv2d(input=im_tf,filter=C_4_tf_0,strides=[1, 1, 1, 1], padding='SAME')
    sess.run(tf.global_variables_initializer())
    im_1,im_2,im_3, im_4=sess.run([ConOut_1,ConOut_2,ConOut_3,ConOut_4])

{% endhighlight %}

Let's see how the image look after the convolutions.


{% highlight ruby %}
%matplotlib inline

plt.subplots(2,2,figsize=(10,10))

plt.subplot(2, 2, 1)
plt.imshow(im_1[0,:])
plt.xlabel(" C_1 = trivial ")

plt.subplot(2, 2, 2)
plt.imshow(im_2[0,:])
plt.xlabel(" C_2 = blur ")

plt.subplot(2, 2, 3)
plt.imshow(im_3[0,:])
plt.xlabel(" C_3 =border ") 

plt.subplot(2, 2, 4)
plt.imshow(im_4[0,:])
plt.xlabel(" C_4= y-direction change ");

{% endhighlight %}


<center>
<img src="{{ '/assets/img/Images_nn_and_tf_5_files/Images_nn_and_tf_5_32_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


In our next post we use convolutions to build our neural network.


{% highlight ruby %}

{% endhighlight %}
