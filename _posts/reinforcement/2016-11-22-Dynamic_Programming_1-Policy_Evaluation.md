---
layout: post
title:  "Dynamic Programing 1: Policy Evaluation"
date:   2016-11-22
category: reinforcement
---



DP refers to a collection of algorithms to compute optiomal assuming a perfect model of the environment. Some of their disadvantages are:

- Assumption of perfect model.
- Computationally expensive.

Our goal is to find good policies so that we satisfy the Bellman optimality equations. 

$$
\begin{align*}
v_*(s)&= \max_a \mathbb{E}_{\pi_*} \left[ R_{t+1}+ \gamma v_*(S_{t+1}) \mid S_t=s, A_t =a \right]\\
&=\max_a \sum_{s',r} p(s',r \mid s,a)[r+\gamma v_*(s')]
\end{align*}
$$

or

$$
\begin{align*}
q_*(s,a)&=\mathbb{E}[R_{t+1}+\gamma \max_{a'}q_*(S_{t+1},a')\mid S_t=a,A_t=a ] \\
& = \sum_{s',r} p(s',r \mid s,a)\left[ r + \gamma \max_{a'} q_*(s',a')\right].
\end{align*}
$$

**From now on we assume that there are only finitely many states.**

# Policy evalution

Firs we need to find how to compute the state-value function $$v_{\pi}$$ for a policy $$\pi$$. Recal that we have the Bellman equation for $$v_{\pi}$$ as 

$$
\begin{align*}
v_{\pi}(s)&= \max_a \mathbb{E}_{\pi} \left[ R_{t+1}+ \gamma v_{\pi}(S_{t+1}) \mid S_t=s, A_t =a \right]\\
&= \sum_{a,s',r} \pi(a\mid s)p(s',r \mid s,a)[r+\gamma v_{\pi}(s')]
\end{align*}
$$

as the state space is finite we can consider a sequence of functions $$v_0,\ldots$$ with $$v_i:\mathcal{S}\to \mathbb{R}$$ (that we can identify with arrays) defined recursively as 

$$
\begin{align*}
v_{k+1}(s)&= \sum_{a,s',r} \pi(a\mid s)p(s',r \mid s,a)[r+\gamma v_{k}(s')]
\end{align*}
$$

We have that $$v_k\to v_{\pi}$$ when $$k\to \infty$$. So this gives an iterative method for finding the state-value function. Before seen this in code we need to make some extra assumption in our enviroments.

We assume that it contains a dictionary env.P for which env.P[$$s',r,s,a$$] represents $$p(s',r \mid s,a)$$.

A list env.rewards, containing the list of distinct possible rewards to be obtained.

*Exercise: Can this be avoid/simplified?*

Let's see this in code. 


{% highlight ruby %}
from collections import defaultdict
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    '''policy: dict. A policy in the form of a matrix of probabilities, with entries [Action,State].
       env: Enviroment.
       discount_factor: the gamma above.
       theta: error allowed.
        '''
    
    states=env.states
    actions=env.actions
    rewards=env.rewards
    #We initiate the states at zero.
    V = defaultdict(int)
    counter=0
    while True:
        V_prev=V.copy()

        for s in states:
            sum_v=0
            for a in actions:
                for s_ in states:
                    for r in rewards:
                        sum_v+=policy[a,s]*env.P[s_,r,s,a]*(r+discount_factor*V_prev[s_])
            V[s]=sum_v
            
        error=0
        for s in states:
            error+=abs(V_prev[s]-V[s])
        
        if error<theta:
            break
            
    return V
{% endhighlight %}

We need a testing ground. For this we will use an playground enviroment. We have created a toy enviroment following example 3.8 from the notes, Gridworld. 

## The gridworld

The following example follows GridWorld from Sutton's Reinforcement Learning book chapter 3, example 3.8. It consists of a 5x5 grid. With two portals that give you points for using them. [ADD DETAILS]. First, we import an enviroment we built for this purpose.


{% highlight ruby %}
import sys
import numpy as np
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv
env = GridworldEnv()
{% endhighlight %}

We can take a quick glance at the enviroment by rendering it.


{% highlight ruby %}
env.render()
{% endhighlight %}

    X a o b o 
    
    o o o o o 
    
    o o o B o 
    
    o o o o o 
    
    o A o o o 
    


Lets take some actions to see how the enviroment changes.


{% highlight ruby %}
print(env.step('DOWN'))
env.render()
print(env.step('RIGHT'))
env.render()
print(env.step('UP'))
env.render()
{% endhighlight %}

    ((0, 1), -1, False, {'Info': 'NoInfo'})
    o a o b o 
    
    X o o o o 
    
    o o o B o 
    
    o o o o o 
    
    o A o o o 
    
    ((1, 1), -1, False, {'Info': 'NoInfo'})
    o a o b o 
    
    o X o o o 
    
    o o o B o 
    
    o o o o o 
    
    o A o o o 
    
    ((1, 0), -1, False, {'Info': 'NoInfo'})
    o X o b o 
    
    o o o o o 
    
    o o o B o 
    
    o o o o o 
    
    o A o o o 
    


In particular we see, that as expected, when we reach $$a$$, we get transported to $$A$$. Let's reset our enviroment.


{% highlight ruby %}
env.reset();
{% endhighlight %}

Now, let's see what happens when we find our policy evaluation function. To check on this we need some policy, so let's try some. Let's start we making equaly probable to move in any direction. If everything is working properly we should get the same results as in the book.


{% highlight ruby %}
pol1={(x,(i,j)):0.25 
      for i in range(5) for j in range(5) for x in [0,1,2,3]}
{% endhighlight %}


{% highlight ruby %}
V=policy_eval(pol1, env, discount_factor=0.90, theta=0.00001)
{% endhighlight %}




And we obtained a value function given by


{% highlight ruby %}
for i in range(5):
    print('|',end="")
    for j in range(5):
        print(str(round(V[j,i],2))+" | ",end="")
    print("\n")

{% endhighlight %}

    |3.31 | 8.79 | 4.43 | 5.32 | 1.49 | 
    
    |1.52 | 2.99 | 2.25 | 1.91 | 0.55 | 
    
    |0.05 | 0.74 | 0.67 | 0.36 | -0.4 | 
    
    |-0.97 | -0.44 | -0.35 | -0.59 | -1.18 | 
    
    |-1.86 | -1.35 | -1.23 | -1.42 | -1.98 | 
    


Which agrees with the course book. Let's try a random policy now.


{% highlight ruby %}
pol2={}
from random import random
for i in range(5):
    for j in range(5):
        A=[random() for i in range(4)]
        s=sum(A)
        A=[a/s for a in A]
        for l in range(4):
            pol2[l,(i,j)]=A[l]
{% endhighlight %}


{% highlight ruby %}
V2=policy_eval(pol2, env, discount_factor=0.90, theta=0.00001)
for i in range(5):
    print('|',end="")
    for j in range(5):
        print(str(round(V2[j,i],2))+" | ",end="")
    print("\n")

{% endhighlight %}

    |0.43 | 8.13 | 3.88 | 4.28 | -0.95 | 
    
    |0.07 | 1.6 | 2.44 | 0.61 | -0.62 | 
    
    |-1.27 | -0.07 | -0.08 | -0.8 | -1.13 | 
    
    |-2.92 | -1.19 | -0.89 | -1.48 | -2.14 | 
    
    |-2.27 | -2.07 | -1.5 | -2.23 | -3.01 | 
    


Let's see what happens if the policy always chooses to move up


{% highlight ruby %}
pol3={}
for i in range(5):
    for j in range(5):
        for l in range(4):
            if l==0:
                pol3[l,(i,j)]=1
            else:
                pol3[l,(i,j)]=0
{% endhighlight %}


{% highlight ruby %}
V3=policy_eval(pol3, env, discount_factor=0.9, theta=0.00001)
for i in range(5):
    print('|',end="")
    for j in range(5):
        print(str(round(V3[j,i],2))+" | ",end="")
    print("\n")

{% endhighlight %}

    |-10.0 | 24.42 | -10.0 | 18.45 | -10.0 | 
    
    |-9.0 | 21.98 | -9.0 | 16.61 | -9.0 | 
    
    |-8.1 | 19.78 | -8.1 | 14.94 | -8.1 | 
    
    |-7.29 | 17.8 | -7.29 | 13.45 | -7.29 | 
    
    |-6.56 | 16.02 | -6.56 | 12.11 | -6.56 | 
    


Another example/exercises can be taken from [Wild Machine Learning](http://www.wildml.com/2016/10/learning-reinforcement-learning/) that in turn follows the example in Sutton's Reinforcement Learning book. It uses the small gridworld enviroment build by [Denny Britz](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/gridworld.py). 
