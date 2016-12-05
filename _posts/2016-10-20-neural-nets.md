---
layout: post

title:  "Neural Networks"

date:   2016-10-19

published: true
---




This notebook and the ones following are design to cover the basics of neural networks in python. 

# Section 1: Introduction

The goal of neural nets is to train neurons, such that the network learn to perform complex operations. In order to made this clear, we need to understand what a neuron is, what are the different models for a neuron, and how to train them.

## 1.1 Models for Neurons.

In essence, neurons are objects that for an input $$(x_1,\ldots,x_n)$$ has an output $$y$$. A training set is a collection of inputs and corresponding outputs, a neuron is trained by the training set by finding the appropriate parameters that defined the neuron. There are several model for neurons. We study some of the basic ones next.

<center><img src="{{ '/assets/img/neural_nets_1_files/image.png' | prepend: site.baseurl }}" alt=""> </center>



### 1.1.1 Linear Neurons

For an input $$(x_1,\ldots,x_n)$$ the linear neurons compute the output $$y$$ by letting  

$$y=b+ w_1 x_1+w_2x_2+\ldots w_nx_n.$$ 

The training as we will see later consists on finding the right weights $$w_1,\ldots,w_n$$ and the bias term $$b$$. It is common to make the problem homogeneous (this will make the training easier). That is to change the problem to a problem were there's no bias term. We achieve this by considering the input to be $$(x_1,\ldots,x_n,1)$$ and then the output is just a linear combination with weights $$w_1,\ldots,w_n,1$$. 

### 1.1.2 Binary Thresholds Neurons

The previous class of neurons are good are good for regression problems, but for classification problems we need something with a discrete output. So consider instead a function depending of an extra parameter $$\theta$$ such that 

$$y=\begin{cases}
1& \text{if }w_1 x_1+w_2x_2+\ldots w_nx_n\geq \theta;\\
0 & \text{otherwise}.
\end{cases}$$

Note that we can also homogenize this question so we may assume $$\theta=0$$.

### 1.1. 3 Rectified Linear Neurons

-In this case the output is obtained as 

$$
\begin{align*}
y&=\begin{cases}
w_1 x_1+w_2x_2+\ldots w_nx_n & \text{if  } w_1 x_1+w_2x_2+\ldots w_nx_n\geq 0;\\
0 & \text{otherwise}.
\end{cases} \\
& \\
&=\max\{0,w_1 x_1+w_2x_2+\ldots w_nx_n\}. 
\end{align*}
$$ 

These neurons have advantages, like improving speed in the training of deep neural networks.

### 1.1.4 Sigmoid Neurons

The sigmoid function ives a smooth output that is always between 0 and 1. 

$$y = \frac{1}{1+e^{-(w_1 x_1+w_2x_2+\ldots w_nx_n)}}.$$

Note what happens when $w_1 x_1+w_2x_2+\ldots w_nx_n\to \pm \infty$. 


### 1.1.5 Stochastic Binary

This function treats the output of the Sigmoid as the probability of an outcome


$$P(Y=1) = \frac{1}{1+e^{-(w_1 x_1+w_2x_2+\ldots w_nx_n)}}.$$


## 1.2 Perceptrons: The network of one neuron.

Perceptrons use Binary Threshold Neurons. Assume that we have some given some training data $$(\boldsymbol{x_i},y_i)$$, where $$\boldsymbol{x_i}$$ represents an $$n$$-tuple and $$$y_i\in \{0,1\}$$. Note that a Binary Threshold Neurons corresponds to a hyperplane, that tries to separate the points with output 1, from the points with output 0. For example, suposse we want to classify the data $$(-1,2)\mapsto 1 $$ $$ (2,0)\mapsto 1$$ $$ (-1,-3)\mapsto 0$$ $$ (1,-2) \mapsto 0$$ The binary threshold neuron with weights $$w_1=-1$$ and $$w_2=1$$, will incorrectly classify the points (fig. 1), meanwhile the binary threshold neuron with weights $$w_1=1$$, $$w_2=1$$ (fig. 2) does classify them correctly.  


{% highlight ruby %}
import numpy as np
import matplotlib.pyplot as plt


plt.figure(figsize=(9,3))

ax1 = plt.subplot(121)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax1.axhline(y=0,color='k')
ax1.axvline(x=0,color='k')
ax1.plot([-1,2],[2,0],'ro',label='Output 1')
ax1.plot([-1,1],[-3,-2],'bo',label='Output 0')
ax1.set_xlim([-4,4])
ax1.set_ylim([-4,4])
ax1.grid(True, which='both')
plt.fill([-4,-4,4],[-4,4,4],color='r',alpha=0.2)
plt.fill([-4,4,4],[-4,-4,4],color='b',alpha=0.2)
plt.title('fig. 1')
ax1.legend()


ax2 = plt.subplot(122)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.axhline(y=0,color='k')
ax2.axvline(x=0,color='k')
ax2.plot([-1,2],[2,0],'ro',label='Output 1')
ax2.plot([-1,1],[-3,-2],'bo',label='Output 0')
ax2.set_xlim([-4,4])
ax2.set_ylim([-4,4])
ax2.grid(True, which='both')
plt.fill([-4,4,4],[4,4,-4],color='r',alpha=0.2)
plt.fill([-4,-4,4],[4,-4,-4],color='b',alpha=0.2)
plt.title('fig. 2')
ax2.legend()


plt.draw()
plt.tight_layout()
plt.show()

{% endhighlight %}

<center><img src="{{ '/assets/img/neural_nets_1_files/neural_nets _1_18_0.png' | prepend: site.baseurl }}" alt=""></center> 

 



So, the natural question is how do we obtain the right weights. 

### 1.2.1 Training the Perceptron 

The algorithm for training the perceptron is a simple one. Choose a random vector of weights $$\boldsymbol{w}=(w_1,\ldots,w_n)$$, then go through the training data points. For every point $$\boldsymbol{x_i}$$ do one of the following:

- If $$\boldsymbol{x_i}$$ is correctly classified, do nothing.
- If $$\boldsymbol{x_i}$$ is incorrectly classified as $$0$$, then make $$\boldsymbol{w}=\boldsymbol{w}+\boldsymbol{x_i}$$.
- If $$\boldsymbol{x_i}$$ is incorrectly classified as $$1$$, then make $$\boldsymbol{w}=\boldsymbol{w}-\boldsymbol{x_i}$$.

Keep repeating this procedure, that is go trought all data points again. After posssibly going trhought the data points several times, we will obtain $$\boldsymbol{w}$$ that separates the ones with output 0 from the ones with output 1. 

At this moment you should be wondering why this works. First: It doesn't always works, because there are sets that can not be classified, for example:


{% highlight ruby %}
plt.figure(figsize=(4,4))

ax1 = plt.subplot(111)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax1.axhline(y=0,color='k')
ax1.axvline(x=0,color='k')
ax1.plot([-1,1],[-1,1],'ro',label='Output 1')
ax1.plot([-1,1],[1,-1],'bo',label='Output 0')
ax1.set_xlim([-4,4])
ax1.set_ylim([-4,4])
ax1.grid(True, which='both')
plt.title("Perceptrons can't always classify")
ax1.legend()

plt.draw()
plt.tight_layout()
plt.show();

{% endhighlight %}

<center><img src="{{ '/assets/img/neural_nets_1_files/neural_nets_1_23_0.png' | prepend: site.baseurl }}" alt=""></center> 



But, if there is a hyperplane separating the points with output 1 from the points with output 0, then this process finishes after finitely many steps. Unfortunately, the convergence tends to be slow. 

## 1.3 Linear Neuron

Given some training data, it is higly improbable that it will fit in linear neuron, so in this case we want is to minimize the error, we make this more precise next. Given some training data $$(\boldsymbol{x_i},y_i)$$ where $$y_i\in \mathbb{R}$$, what is the vector $$\boldsymbol{w}=(w_1,\ldots,w_n)$$ for which 

$$\frac{1}{2}\sum_i(y_i-\boldsymbol{w} \cdot \boldsymbol{x_i})^2=\text{Error}$$

is as small as possible. If we follow the analytic path we obtain the usual linear regresssion, instead we want to use something different, that will generalize to larger neuran networks.

## 1.3.1 Linear Neuron Training (Full Batch)

One algorithmic way to get to the min of a function is via gradient descent. We start with some initial weight vector $$\boldsymbol{w_0}$$, we choose a learning rate $$\epsilon$$ and we set 

$$ \boldsymbol{w_{t+1}}=\boldsymbol{w_{t}}+\Delta \boldsymbol{w_t}=\boldsymbol{w_{t}}-\epsilon \sum_i \boldsymbol{x_i}(y_i- \boldsymbol{w_t}\cdot \boldsymbol{x_i}) $$

Choosing the learning $\epsilon$ small enough the sequence $\boldsymbol{w_t}$ converges to the analytic solution. This procedure has several shortcomings, one of them being that it relies in knowing all the data. Another approach is to use every new data to update the weight vector. We do this next.

## 1.3.2 Linear Neuron Training (Online)

We ramdonly pick a training data element $$(\boldsymbol{x_i},y_i)$$ and put it back, we use this element to update the weight vector as 

$$ \boldsymbol{w_{t+1}}=\boldsymbol{w_{t}}+\Delta \boldsymbol{w_t}=\boldsymbol{w_{t}}-\epsilon \boldsymbol{x_i}(y_i- \boldsymbol{w_t}\cdot \boldsymbol{x_i}) $$

This procedure has an extremely slow convergence rate (it zigzags towards the minimum). 

One way to fix the shortcomming of both techniques is to use mini-batches, take random samples of small size and update using the formula for Full Batch.

## 1.4 Logistic Neurons

This is just another name for Sigmoid Neurons. Recall that in this case the output is computed as

$$y = \frac{1}{1+e^{-(w_1 x_1+w_2x_2+\ldots w_nx_n)}}.$$
and we want to find the right weights $w_1,\ldots,w_n$. As we the linear case we want to use gradient descent to minimize

$$ \text{Error} = \frac{1}{2} \sum_i\left(y_i-\frac{1}{1+e^{-\boldsymbol{w}\cdot \boldsymbol{x_i}}}\right)^2.$$

And so the updating process  would go as

$$ \boldsymbol{w_{t+1}}=\boldsymbol{w_{t}}+\Delta \boldsymbol{w_t}=\boldsymbol{w_{t}}- \epsilon\sum_i \boldsymbol{x_i} \left( \frac{1}{1+e^{-\boldsymbol{w}\cdot \boldsymbol{x_i}}} \right)\left( 1-\frac{1}{1+e^{-\boldsymbol{w}\cdot \boldsymbol{x_i}}} \right) \left(y_i- \frac{1}{1+e^{-\boldsymbol{w}\cdot \boldsymbol{x_i}}} \right) $$

As above we could use a on-line training or a mini-batch, we leave the details to the reader.

# 2. Neural Networks

We move now to more complex systems. In general a neural network consists of a collection of neurons interconnected. The first kind of network we will study is that for whih each neuron has an activation function $$y=f(x_1,\ldots,x_n)$$ that is smooth. For example, if $$f(x)$$ is the sigmoid. Our goal is to ind the "best" weights for each of the neurons. 

## 2.1 Backpropagation

We assume that the neural network is feedforward, this basically means that there are not cycles.

<center><img src="{{ '/assets/img/neural_nets_1_files/image.png' | prepend: site.baseurl }}" alt=""> </center>


Back propagation is a technique to compute the local gradient efficiently, that is by how much the weights in each unit must change. The general explanation requires some (annoying) notation, in order to make the ideas clear we consider a smaller case and let the reader imaging the general case. Consider the network given by the next figure.

<center> <img src="{{ '/assets/img/neural_nets_1_files/image.png' | prepend: site.baseurl }}" alt=""> </center>


In this case we have on output unit, one hidden layer with two units, and one input. Our goal is to minimize the error $$E=E(w_{1},w_{2},w_{31},w_{32})$$ where $$w_{ij}$$ is the $$j$$-th weight of the $$i$$-th unit, we minimize $$E$$ using gradient descend so we need to find $$\frac{ \partial E}{\partial w_{ij}}$$. Let 

$$\varphi(z)=\frac{1}{1+e^{-z}},$$ 

and $$f_i$$ the sigmoid activation function associated to the $$i$$-th unit, that is $$f_1(x)=\varphi(w_1x)$$, $$f_2(x)=\varphi(w_2x)$$, and $$f_3(a,b)=\varphi(w_{31}a+w_{32}b)$$.

Note that given a data point $$(x_1,x_2,y)$$ the error function is given by 

$$E=\frac{1}{2}\big (\varphi \big( w_{31}f_1(x)+w_{32}f_2(x) \big)-y \big )^2$$

Now, a trivial computation shows $$\frac{\partial \varphi(z)}{ \partial z }=\varphi(z)(1-\varphi(z))$$, and we can use the chain rule to get

$$ 
\begin{multline} 
\frac{\partial E}{\partial w_{3i}}= \big (\varphi \big( w_{31}f_1(x)+w_{32}f_2(x) \big)-y \big ) \cdot \big (1-\varphi \big( w_{31}f_1(x)+w_{32}f_2(x) \big)\big ) \\
 \cdot \varphi \big( w_{31}f_1(x)+w_{32}f_2(x) \big) \cdot f_i(x) 
\end{multline}
$$

Meanwhile, 

$$ 
\begin{multline} 
\frac{\partial E}{\partial w_{i}}= \big (\varphi \big( w_{31}f_1(x)+w_{32}f_2(x) \big)-y \big ) \cdot \big (1-\varphi \big( w_{31}f_1(x)+w_{32}f_2(x) \big)\big )  \\ 
  \cdot \varphi \big( w_{31}f_1(x)+w_{32}f_2(x) \big)\cdot w_{3i}\cdot f_i(x)\cdot (1-f_i(x))\cdot x  
\end{multline}
$$


That is $$ \frac{\partial E}{\partial w_{i}} =w_{3i}\cdot x\cdot(1-f_i(x)) \cdot \frac{\partial E}{\partial w_{3i}}$$
and we can compute the earlier errors from the later ones. We can use this, plus the ideas above, to reach a minimum, unfortunately we can only guarantee that this is a local minimum. 


{% highlight ruby %}

{% endhighlight %}
