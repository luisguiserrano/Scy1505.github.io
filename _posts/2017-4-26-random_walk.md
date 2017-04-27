---
layout: post
title:  "Random walk of molecules"
date:   2017-04-26
category: general
---

He says in cool voice: "Soooo... I am reading this excellent book called [The Machinery of Life, by David S. Goodsell](https://www.amazon.com/Machinery-Life-David-S-Goodsell/dp/0387849246), in an attempt to learn some basic Molecular Biology. And surprise, surprise! Page 10 has a claim about molecules walking randomly are bound to hit each other, and naturally, I decided to model it."  

# The question

Suppose that there are two molecules, $$A$$ and $$B$$, inside a compartment of a cell (think mitochondria or the cell itself), most molecules move by bouncing back and forth against water molecules which creates a Brownian movement. Will the molecules $$A$$ and $$B$$ eventually collide? If the molecules were to be huge relatively to the cell, you probably would answer yes right away, but the molecules are tiny. As a comparison, imagine the cell as to be a blank piece of paper, and the molecules are two dots on it moving randomly, will they collide? I was surprised a first to find the answer to this question to be yes, they do. But after a more careful thinking, I came up with a simple explanation of way this happens. 

Consider a finite set of points inside the cell like in the figure such that a given point the smaller molecule has to contain at least of the points, see the figure.


{% highlight ruby %}
fig = plt.figure(figsize=(5, 5), dpi=100)
plt.plot(points_x,points_y,'ro',ms=0.5)
circle = plt.Circle((0.21, 0.21), 0.05, color='b')
plt.gca().add_artist(circle)
circle = plt.Circle((0.61, 0.71), 0.03, color='g')
plt.gca().add_artist(circle)
plt.show()
{% endhighlight %}


<center>
<img src="{{ '/assets/img/random_walk_files/random_walk_3_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


Note that a collision happens if both molecules contain the same point. As the movement is Brownian the probability of the molecule $$A$$ to contain the point $$i$$ at a time $$t$$ sufficiently large is 

$$P(i \in A\mid_{T=t})>0,$$ 

similarly $$P(i \in B\mid_{T=t})>0$$ and as the movements are independent of each other, under our assumptions, $$P(i \in A\cap B\mid_{T=t})>0$$ but we also know that

$$P(i \in A\cap B\mid_{T=t})=P(i \in A\cap B\mid_{T=t'})>0$$ 

for $$t,t'$$ sufficiently large, hence the law of large numbers implies that this eventually happens. 

Now the goal of this post is to show how we can model this on Python. 


## The model

We start by creating a class that represents the molecules, and we allow ourselves to give the class an initial position, a rate of movement, and a size. Later, we give these parameters values that would mimic the real life situation.

Our model for a protein is the same as for a two-dimensional cow, that is a circle, and for the cell, we use a square of side 1. 

Of importance to us are the methods update and collide, the first one updates the position of the molecule after an instant of time by choosing a random direction if the molecule 'hits' the wall then it bounces. The second one determines if a collision has occurred with another molecule. 


{% highlight ruby %}
import numpy as np

class molecule:
    
    def __init__(self,pos,rate,size):
        self.pos=pos
        self.rate=rate
        self.size=size
        
    def update(self):
        #Selects a random direction vector
        v=np.random.multivariate_normal(np.array([0,0]), [[1,0],[0,1]] )
        
        #The possition where the molecule would be if not wall 
        possi_pos= np.abs(self.pos+self.rate*v)
        
        #Corrects the position in case the molecule hit the wall
        while possi_pos[0]>1 or possi_pos[1]>1 or possi_pos[0]<0 or possi_pos[1]<0:
            if possi_pos[0]>1:
                possi_pos[0]=2-possi_pos[0]
            elif possi_pos[1]>1:
                possi_pos[1]=2-possi_pos[1]
            elif possi_pos[0]<0:
                possi_pos[0]=np.abs(possi_pos[0])
            elif possi_pos[1]<0:
                possi_pos[1]=np.abs(possi_pos[1])
        
        #updates postion
        self.pos=possi_pos
        
    def colide(self,other):
        return np.linalg.norm(self.pos-other.pos)<self.size+other.size
{% endhighlight %}

We can try it out, let's create two molecules moving at a speed of .3 units of distance per unit of time and size one thousandth of the size of the cell.


{% highlight ruby %}
A=molecule(np.array([0.5,0.5]),0.3,0.001)
B=molecule(np.array([0.2,0.2]),0.3,0.001)
{% endhighlight %}

Now, let's update their position until they collide or a million units of time have passed. We keep their positions in a variable so we can graph it later.


{% highlight ruby %}
np.random.seed(13)
t=0
posA=[]
posB=[]
while t<1000000:
    t+=1
    posA+=[A.pos]
    posB+=[B.pos]
    A.update()
    B.update()
    if A.colide(B):
        print(t)
        break
{% endhighlight %}

    17257


So, it took $$17257$$ units of time for the collision to happen. That sounds like a lot, but is it? Here are some facts. A molecule is about 1/1000 the size of a cell; it travels its size per nanosecond (they are fast!). So if our time unit is microsecond, $$10^{-6}$$ of a second, then their speed is about 1 cell units per microsecond. We are assuming 0.3 cell units per microsecond, so our model is actually slower, more interesting it took 17257 microseconds for the collision to happen, that is 0.000017257 seconds for it to happen, that is incredibly fast!

What about a graph, well, this is tricky, so here is what happens every millisecond. This graph could be improved by either giving a color scheme to the time or creating an animation, let's call it homework.


{% highlight ruby %}
import matplotlib.pyplot as plt

plt.plot([a[0] for a in posA[::1000]],[a[1] for a in posA[::1000]])
plt.plot([a[0] for a in posA[::1000]],[a[1] for a in posB[::1000]])

plt.show()
{% endhighlight %}


<center>
<img src="{{ '/assets/img/random_walk_files/random_walk_14_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


Finally, it is an interesting question, raised by my smart wife, about what is the probability distribution of the collision time. We run our model 100 times and keep the results to see what happens. (This will take a while)


{% highlight ruby %}
t=0
times=[]
A=molecule(np.array([0.99,0.99]),0.3,0.001)
B=molecule(np.array([0.01,0.01]),0.3,0.001)

for i in range(100):
    t=0
    while t<1000000:
        t+=1
        A.update()
        B.update()
        if A.colide(B):
            times+=[t]
            break
    
{% endhighlight %}

we break it into 50 intervals and find how often each interval happens.


{% highlight ruby %}
max_time=max(times)
time_frequency=[0]*51
for time in times:
    i=(50*time)/max_time 
    time_frequency[int(i)]+=1
{% endhighlight %}

and we get a graph


{% highlight ruby %}
fig = plt.figure(figsize=(10, 10), dpi=100)
plt.bar([x for x in range(51)],time_frequency)
plt.xticks([x for x in range(0,51,10)],[x*max_time//50000 for x in range(0,51,10)])
plt.xlabel("Time in miliseconds")
plt.show()
{% endhighlight %}


<center>
<img src="{{ '/assets/img/random_walk_files/random_walk_20_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


Which suggests a Poisson distribution.