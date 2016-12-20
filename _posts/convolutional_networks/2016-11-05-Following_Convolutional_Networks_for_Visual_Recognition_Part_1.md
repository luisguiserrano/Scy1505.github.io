---
layout: post
title:  "Following Convolutional Networks for Visual Recognition - Part 1"
date:   2016-11-05
category: convolutional_networks
---

The goal of this and the following notebooks is to expand on the content of the Course of Convolutional Networks for Visual Recognition found [here](http://cs231n.github.io/). 

I will assume that you have read [Image Classification: Data-driven Approach, k-Nearest Neighbor, train/val/test splits](http://cs231n.github.io/classification/), as well as [A few useful things to know about Machine Learning](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf). We expand on the details here.

# Review concepts

**Classiffier:** The technique(s) being used to solve the classification problem.

**Train/Test splits:** The data should be split into a subset for training and another for testing. Never to touch the testing one in order to improve performance.

**Overfitting:** When the classifier is accuarate with the training data, but perfoms badly with the testing data, it increases the variance of the error. Common in decession tree classifiers. 

**Bias:** An inner error of the training. Common when underfitting. It represents a mean in the error.


**Validation/Cross Validations:** Techniques used to tune the (meta) parameters associated with the classifier.

**The curse of dimentionality:** Phrase coined to explain the fact that generalizing becomes exponentially harder as the number of dimension increase. 


# $$K$$-Nearest Neighbor in sklearn

This algorithm takes no time training but a lot of time testing. In Sklearn documentation, [here](http://scikit-learn.org/stable/modules/neighbors.html), there are the details of the use. We summarize the main points next.

The classifier can be imported as:


{% highlight ruby %}
from sklearn.neighbors import KNeighborsClassifier

# We also import numpy because we needed for our example.
import numpy as np
{% endhighlight %}

The classifier has the following parameters:

(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)

The explanation can be found [here](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier). 

For us the important part is the the parameter given by algorithm, there are four choices ‘auto’, ‘ball_tree’, ‘kd_tree’, and ‘brute’.


## 'auto'

It attempts to decide the most appropriate algorithm based on the values passed to fit method.

## 'brute'

It uses the brute force approach of comparing with every element in the training data. This is computationally expensive at testing, running at order $$\mathcal{O}(d \cdot n^2)$$. Where there are $$n$$ elements in a $$d$$-dimensional space as the training data. 

## 'kd_tree'

Improves efficiency by reducing the complexity of the brute force approach. The idea is that a point that if two points $$a$$, $$b$$ are far from each other, and $$c$$ is close to $$b$$ then $$a$$ is far from $$c$$. This is achieve by creating a niary tree structure to encode this information. It runs with order $$\mathcal{O}(d\cdot n\cdot \log(n))$$. But in practice is performance reduces when $$d>20$$.

## 'ball_tree'

Improves over 'kd_tree' for larger dimensions.

# Hands on

We play with the [CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html), to compare the speed of the $$k$$-neigbor classifier under this four algorithms. We first get the pictures from the picle files.


{% highlight ruby %}
def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo,encoding='latin1')
    fo.close()
    return dict

# unplickles the training data 
data_batch=[0]*5
data_batch[0]=unpickle('CIFAR-10/data_batch_1')
data_batch[1]=unpickle('CIFAR-10/data_batch_2')
data_batch[2]=unpickle('CIFAR-10/data_batch_3')
data_batch[3]=unpickle('CIFAR-10/data_batch_4')
data_batch[4]=unpickle('CIFAR-10/data_batch_5')

#puts the training data together
X_train=np.concatenate(tuple(data_batch[i]['data'] for i in range(5)))
y_train=np.concatenate(tuple(data_batch[i]['labels'] for i in range(5)))


# unpickles the test data and puts it in a dictionary call test
test=unpickle('CIFAR-10/test_batch') 
X_test=test['data'].copy()
y_test=test['labels']
{% endhighlight %}

We can create our different classifiers:


{% highlight ruby %}
brute_Cl=KNeighborsClassifier(algorithm='brute')
kd_tree_Cl=KNeighborsClassifier(algorithm='kd_tree')
ball_tree_Cl=KNeighborsClassifier(algorithm='ball_tree')
{% endhighlight %}

We can feed the data to train our models, and time how long it takes.


{% highlight ruby %}
import time

start_time=time.time()
brute_Cl.fit(X_train,y_train)
brute_train_time=time.time()-start_time
print("Training the brute force takes "+str(round(brute_train_time,4))+" seconds.")


start_time=time.time()
kd_tree_Cl.fit(X_train,y_train)
kd_tree_train_time=time.time()-start_time
print("Training the kd_tree takes "+str(round(kd_tree_train_time,4))+" seconds.")

start_time=time.time()
ball_tree_Cl.fit(X_train,y_train)
ball_tree_train_time=time.time()-start_time
print("Training the ball_tree takes "+str(round(ball_tree_train_time,4))+" seconds.")
{% endhighlight %}

    Training the brute force takes 0.0076 seconds.
    Training the kd_tree takes 21.145 seconds.
    Training the ball_tree takes 18.7129 seconds.


Now that we have our models trained, we can check how long does it take to predict on the testing data.


{% highlight ruby %}
start_time=time.time()
y_brute=brute_Cl.predict(X_test)
brute_predict_time=time.time()-start_time
print("Testing by brute force takes "+str(round(brute_predict_time,4))+" seconds.")

start_time=time.time()
y_kd_tree=kd_tree_Cl.predict(X_test)
kd_tree_predict_time=time.time()-start_time
print("Testing by kd_tree takes "+str(round(kd_tree_predict_time,4))+" seconds.")

start_time=time.time()
y_ball_tree=ball_tree_Cl.predict(X_test)
ball_tree_predict_time=time.time()-start_time
print("Testing by brute force takes "+str(round(ball_tree_predict_time,4))+" seconds.")

{% endhighlight %}

    Testing by brute force takes 78.5795 seconds.
    Testing by kd_tree takes 2581.853 seconds.
    Testing by brute force takes 2051.1635 seconds.


We can check the accuracy by 


{% highlight ruby %}
brute_acc=np.mean(y_brute==y_test)
print("Brute force gives an accuracy of "+str(brute_acc)+".")

kd_tree_acc=np.mean(y_kd_tree==y_test)
print("kd_tree gives an accuracy of "+str(kd_tree_acc)+".")

ball_tree_acc=np.mean(y_ball_tree==y_test)
print("Ball_tree gives an accuracy of "+str(ball_tree_acc)+".")
{% endhighlight %}

    Brute force gives an accuracy of 0.3398.
    kd_tree gives an accuracy of 0.3398.
    Ball_tree gives an accuracy of 0.3398.


Which is expected. You may be wondering why does it take so long to train and test in the kd_tree and ball_tree cases, the reason is the dimension; $$32\times 32\times 3$$ in our case. For larger dimensions this is not a faster process than doing brute force.

# Preprocessing in Sklearn

Maybe, we could speed up the proccess by preprocessing. This process is fairly common and, in some cases, necessary. The most common preprocessing technique consists on removing the mean and scaling by the inverse of the standard deviation. That is, making the data have a mean of 0 and a standard deviation of 1. Sklearn provides a [useful tool](http://scikit-learn.org/stable/modules/preprocessing.html) for this.


{% highlight ruby %}
from sklearn.preprocessing import StandardScaler
{% endhighlight %}

The StandardScaler utility saves the information of the mean and standard deviation so it can be used for processing the test data as well. 


{% highlight ruby %}
scaler = StandardScaler().fit(X_train)
X_sca_train=scaler.transform(X_train);
{% endhighlight %}

    /home/felipe/anaconda3/lib/python3.5/site-packages/sklearn/utils/validation.py:429: DataConversionWarning: Data with input dtype uint8 was converted to float64 by StandardScaler.
      warnings.warn(msg, _DataConversionWarning)


An you may be curious to see how an image look after removing this process, I know I am.


{% highlight ruby %}
from matplotlib import pyplot as plt
%matplotlib inline

fig=plt.figure()

ax00 = plt.subplot2grid((2,2), (0,0))
A=np.swapaxes(np.reshape(X_train[0],(3,32,32)).T,0,1)
plt.imshow(A)


ax01 = plt.subplot2grid((2,2), (0,1))
A=np.swapaxes(np.reshape(X_sca_train[0],(3,32,32)).T,0,1)
plt.imshow(A)

ax10 = plt.subplot2grid((2,2), (1,0))
A=np.swapaxes(np.reshape(X_train[13],(3,32,32)).T,0,1)
plt.imshow(A)


ax11 = plt.subplot2grid((2,2), (1,1))
A=np.swapaxes(np.reshape(X_sca_train[13],(3,32,32)).T,0,1)
plt.imshow(A)

plt.show()
{% endhighlight %}


<center>
<img src="{{ '/assets/img/Following_Convolutional_Networks_for_Visual_Recognition_Part_1_files/Following_Convolutional_Networks_for_Visual_Recognition_Part_1_33_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


Now, let's check if it makes a significant change in training or predicting.


{% highlight ruby %}
import time

start_time=time.time()
brute_Cl.fit(X_sca_train,y_train)
brute_train_time=time.time()-start_time
print("Training the brute force takes "+str(round(brute_train_time,4))+" seconds.")


start_time=time.time()
kd_tree_Cl.fit(X_sca_train,y_train)
kd_tree_train_time=time.time()-start_time
print("Training the kd_tree takes "+str(round(kd_tree_train_time,4))+" seconds.")

start_time=time.time()
ball_tree_Cl.fit(X_sca_train,y_train)
ball_tree_train_time=time.time()-start_time
print("Training the ball_tree takes "+str(round(ball_tree_train_time,4))+" seconds.")
{% endhighlight %}

    Training the brute force takes 0.1932 seconds.
    Training the kd_tree takes 15.9176 seconds.
    Training the ball_tree takes 17.1215 seconds.


This is about the same, how about for predicting. (Let's just test five cases). First we need to scale the test data.


{% highlight ruby %}
X_sca_test=scaler.transform(X_test)

start_time=time.time()
y_brute=brute_Cl.predict(X_sca_test[:50])
brute_predict_time=time.time()-start_time
print("Testing by brute force takes "+str(round(brute_predict_time,4))+" seconds.")

start_time=time.time()
y_kd_tree=kd_tree_Cl.predict(X_sca_test[:50])
kd_tree_predict_time=time.time()-start_time
print("Testing by kd_tree takes "+str(round(kd_tree_predict_time,4))+" seconds.")

start_time=time.time()
y_ball_tree=ball_tree_Cl.predict(X_sca_test[:50])
ball_tree_predict_time=time.time()-start_time
print("Testing by brute force takes "+str(round(ball_tree_predict_time,4))+" seconds.")

{% endhighlight %}

    /home/felipe/anaconda3/lib/python3.5/site-packages/sklearn/utils/validation.py:429: DataConversionWarning: Data with input dtype uint8 was converted to float64 by StandardScaler.
      warnings.warn(msg, _DataConversionWarning)


    Testing by brute force takes 0.4914 seconds.
    Testing by kd_tree takes 12.6376 seconds.
    Testing by brute force takes 10.3443 seconds.


So, how long will this take in without scaling?


{% highlight ruby %}

start_time=time.time()
y_brute=brute_Cl.predict(X_test[:50])
brute_predict_time=time.time()-start_time
print("Testing by brute force takes "+str(round(brute_predict_time,4))+" seconds.")

start_time=time.time()
y_kd_tree=kd_tree_Cl.predict(X_test[:50])
kd_tree_predict_time=time.time()-start_time
print("Testing by kd_tree takes "+str(round(kd_tree_predict_time,4))+" seconds.")

start_time=time.time()
y_ball_tree=ball_tree_Cl.predict(X_test[:50])
ball_tree_predict_time=time.time()-start_time
print("Testing by brute force takes "+str(round(ball_tree_predict_time,4))+" seconds.")

{% endhighlight %}

    Testing by brute force takes 0.5111 seconds.
    Testing by kd_tree takes 7.2683 seconds.
    Testing by brute force takes 2.4242 seconds.


So there's not real benefit for this data set to do preprocessing! This is important, since it helps us keep in mind that many of this techniques are dependent of the data and it is better to understand it than applying the techniques blindly. Here is a question for you, why do the processing times worsen when normalizing the data and doing the kd_tree and ball_tree algorithms? (*Hint: findout what the algorithms do*).
