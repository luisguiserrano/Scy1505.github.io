---
layout: post
title:  "NN for image reccognition - Part 1"
date:   2016-12-15
category: NN_for_image_reccognition
---

This is part 1 of our blogpost related with images and tensorflow. The posts follow the following:

1. **Getting the Data.**
2. k-neareast Neighbor.
3. Logistic Regression.
4. A two layer Neural Network.
5. Convolutions in Tensorflow.
6. Convolutional Networks.
7. What's next?

# 1. Getting the Data

The file *aux.py* contains the code for imputing the data under the method *input*. We explain this code next.  

The CIFAR10 database is at [CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html). You can download it by going to the webpage and use the usual means (You could also use python code, but let's keep it as simple as possible). The data comes in a binary file, we can access the info using the pickle package.


{% highlight ruby %}
import pickle
import numpy as np

def unpickle(file):
    """Unpickles the file and returns the dictionary associated to it
    Args:
        x: pickle file
    Returns:
        dict 
    """
    fo = open(file, 'rb')
    dict = pickle.load(fo,encoding='latin1')
    fo.close()
    return dict
{% endhighlight %}

We now read the data, and return 4 numpy arrays: Training features,training labels, test features, and test labels.


{% highlight ruby %}
def _input():
   

    # unplickles the training data 
    data_batch=[0]*5
    data_batch[0]=unpickle('CIFAR-10/data_batch_1.bin')
    data_batch[1]=unpickle('CIFAR-10/data_batch_2.bin')
    data_batch[2]=unpickle('CIFAR-10/data_batch_3.bin')
    data_batch[3]=unpickle('CIFAR-10/data_batch_4.bin')
    data_batch[4]=unpickle('CIFAR-10/data_batch_5.bin')

    #puts the training data together
    X_train=np.concatenate(tuple(data_batch[i]['data'] for i in range(5)))
    y_train=np.concatenate(tuple(data_batch[i]['labels'] for i in range(5)))


    # unpickles the test data and puts it in a dictionary call test
    test=unpickle('CIFAR-10/test_batch.bin') 
    X_test=test['data']
    y_test=test['labels']
    
    return X_train, y_train, X_test, y_test
{% endhighlight %}

Let's check the data that comes form *_input()*  


{% highlight ruby %}
X_train,y_train,X_test,y_test=_input()
{% endhighlight %}


{% highlight ruby %}
print("The first element of the training data looks like: ",X_train[0]," with shape: ", X_train[0].shape)

{% endhighlight %}

    The first element of the training data looks like:  [ 59  43  50 ..., 140  84  72]  with shape:  (3072,)


In order to visualize this better, we need to transform the vector into a 32x32x3 np array with float entries. So we wrap the previous function with something that does this for us:


{% highlight ruby %}
def input(flat=False):
    
    # Get the data
    X_train,y_train,X_test,y_test=_input()
    
    # Make it a float between 0 and 1.
    X_train=np.array(X_train,float)/255.0
    X_test=np.array(X_test,float)/255.0
    
    # Sometimes it is convinient to have the data as a flat array.
    if flat:
        return X_train,y_train,X_test,y_test
    
    # Reshape it to 32x32x3
    X_train = np.reshape(X_train,[-1,3,32,32])
    X_test = np.reshape(X_test,[-1,3,32,32])
    
    # The chanels should be last coordinate, so we transpose
    X_train=X_train.transpose([0,2,3,1])
    X_test=X_test.transpose([0,2,3,1])
    
    return X_train,y_train,X_test,y_test
{% endhighlight %}

Let's also check the labels. They come as integers


{% highlight ruby %}
y_test[0]
{% endhighlight %}




    3



In order to access the true label, we need to know what that is, we do it by brute force:


{% highlight ruby %}
def label(x):
    labels=['Airplane','Automovile','bird','cat','deer','dog','frog','horse','ship','truck']
    return labels[x]
{% endhighlight %}

Let's just do some sanity check


{% highlight ruby %}
X_train,y_train,X_test,y_test = input()
{% endhighlight %}


{% highlight ruby %}
from itertools import product
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 8))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

for i,j in product(range(3),range(3)):
    #print(i*3+j)
    ax = plt.subplot2grid((3,3), (i,j))
    ax.set_title(label(y_train[i*3+j]))
    ax.imshow(X_train[i*3+j])
plt.show()                
{% endhighlight %}


<center>
<img src="{{ '/assets/img/Images_nn_and_tf_1_files/Images_nn_and_tf_1_18_0.png' | prepend: site.baseurl }}" alt=""> 
</center>



{% highlight ruby %}

{% endhighlight %}
