---
layout: post
title:  "The K-armed Bandit"
date:   2016-11-02
category: reinforcement
---


The $$K$$-armed bandit game has many variants, but all of them reduce to making a choice among $$k$$-different ones. More precesily, suppose that you have $$k$$ doors, and behind each door there is a rewad. Assume that the expected reward after choosing the door $$a$$ is given by $$q_*(a)=\mathbb{E}(R|_{A=a}),$$ and that you don't know $$q_*$$ nor the probability distribution.
Now suppose that you play this game several times. How do we maximize our total reward?

To do this, we need to create a function that keeps track of our expectation at step $$t$$, that is, we want a function $$Q_t(a)\approx q_*(a)$$. 

## $$\epsilon$$-greedy

A natural question is how to compute $$Q_t(a)$$, one simple way is to make 

$$Q_t(a)=\frac{\sum_{i=1}^{t-1}R_i\cdot \mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i=a}}.$$ 

That is, the average reward obtained when selecting the door $$a$$. 

The greedy approach would be to always selectthe door with larger expected reward $$Q_t(a)$$. But this clearly won't get you the largest reward in every case. A better alternative is to explore the other doors. So suppose that at every step you flip a (no necessarily fair) coin, and we probability $$\epsilon$$ (small) you decide to explore a random door, or go for the greedy approach. By the laws of large numbers we will have that $$Q_t(a)\to q_*(a)$$when $$t\to \infty$$, and that we probability $$(1-\epsilon)$$, at each step, we will always choose the largest reward door. But how fast do we begging selecting this door? We model this next.

First, we select $$k$$ doors, we make each $$q_*(a)$$ to be given by a ramdon normal distributions of mean variance 1, and means between 10 and 20, and say we are opening doors 200 times. 


{% highlight ruby %}
import numpy as np
k=10
steps=2000
np.random.seed(8)
R=[np.random.normal(np.random.randint(10,21),1,steps) for a in range(k)]
{% endhighlight %}

We can plot this.


{% highlight ruby %}
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure()
plt.boxplot(R, notch=True, patch_artist=True);
plt.show()
{% endhighlight %}

<center>
<img src="{{ '/assets/img/K-armed_Bandit-epsilon_greedy_files/K-armed_Bandit-epsilon_greedy_8_0.png' | prepend: site.baseurl }}" alt=""> 
</center>



As we don't know the rewards behind the door, we can initialize $$Q_t$$ at 15, and the number of times each door has been visited, that is 


{% highlight ruby %}
Q_t=[10]*k
vis=[1]*k
{% endhighlight %}

We do $$\epsilon$$-greedy now, we need an update function funtionfor $$Q_t$$, and let's recover the average reward.


{% highlight ruby %}
def e_greedy(eps=0.0):
    
    #Initialization 
    Q_t=[15]*k
    vis=[1]*k
    
    #Coin flippling
    def flip():
        return  np.random.random() < 1-eps
    
    Rev=[]
    RevAve=[15]
    
    for i in range(steps):
        if flip():
            ma=max(Q_t)
            door=Q_t.index(ma)
            Rev+=[R[door][i]]
            vis[door]+=1
            Q_t[door]=(Q_t[door]*(vis[door]-1)+Rev[-1])/vis[door]
            RevAve+=[((RevAve[-1])*(i+1)+Rev[-1])/(i+2)]
            #print(ma,door,Rev,vis[door],Q_t)
        else:
            ma=np.random.choice(Q_t)
            door=Q_t.index(ma)
            Rev+=[R[door][i]]
            vis[door]+=1
            Q_t[door]=(Q_t[door]*(vis[door]-1)+Rev[-1])/vis[door]
            RevAve+=[((RevAve[-1])*(i+1)+Rev[-1])/(i+2)]
    return RevAve
{% endhighlight %}

We can graph this


{% highlight ruby %}
plt.plot(e_greedy(),label='greedy')
plt.plot(e_greedy(0.2), label='epsilon=0.1')
plt.plot(e_greedy(0.01), label='epsilon=0.01')
plt.legend(loc='best')
#plt.tight_layout()
plt.show();

{% endhighlight %}

<center>
<img src="{{ '/assets/img/K-armed_Bandit-epsilon_greedy_files/K-armed_Bandit-epsilon_greedy_14_0.png' | prepend: site.baseurl }}" alt=""> 
</center>

Note, how the greedy approach got stuck in a local max, this is typical. Once we had the exploring option the reward increases as soon as more doors are explored, it seems that epsilon=0.1 reaches a larger reward, but this is not true in the longer run.


{% highlight ruby %}
k=10
steps=10000
np.random.seed(8)
R=[np.random.normal(np.random.randint(10,21),1,steps) for a in range(k)]
plt.plot(e_greedy(),label='greedy')
plt.plot(e_greedy(0.2), label='epsilon=0.1')
plt.plot(e_greedy(0.01), label='epsilon=0.01')
plt.legend(loc='best')
#plt.tight_layout()
plt.show();
{% endhighlight %}

<center>
<img src="{{ '/assets/img/K-armed_Bandit-epsilon_greedy_files/K-armed_Bandit-epsilon_greedy_16_0.png' | prepend: site.baseurl }}" alt=""> 
</center>