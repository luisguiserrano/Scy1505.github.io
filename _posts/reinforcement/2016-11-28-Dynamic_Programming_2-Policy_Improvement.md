---
layout: post
title:  "Dynamic Programing 2: Policy Improvement"
date:   2016-11-22
category: reinforcement
---



In our previous post we built a policy evaluation function. In this one we will find how to improve a given policy, the algorithm is based on a simple mathematical fact known as the **policy improvement theorem**.

**Theorem:**  Given two deterministic policies $$\pi$$ and $$\pi'$$ such that for all states $$s$$ we have 

$$q_{\pi}(s,\pi'(s))\geq v_{\pi}(s)$$

then $$\pi' \geq \pi$$. Furthermore, if the inequality is strict for some $$s$$ then $$\pi' > \pi$$. 

In essence, this says that if a policy $$\pi'$$ is such that choosing the actions from that policy give better gain that the expected one then the policy is better.

# Making it code

We can use this idea, plus a greedy approach to obtain a better policy, let $$\pi'$$ be given by 

$$ \begin{align*} 
\pi'(s)&=\arg \max_{a} q_{\pi}(s,a)\\ 
&= \arg \max_{a} \sum_{s',r}p(s,r|s,a)\left[ r + \gamma v_{\pi}(s') \right]
\end{align*}
$$

That is, the improved policy takes teh action that looks the best in the short term. 

It is a fact that after a finite number of improvements the procedure reaches an optional policy. We implement the algorithm next. We divide it into three parts, first we implement a policy evaluation almost like the one in the previous post. The difference consists that this version is designed for deterministic policies. 


{% highlight ruby %}
from collections import defaultdict

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    '''policy: (deterministic) dict. A policy in the form of policy[state]=action.
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
            for s_ in states:
                for r in rewards:
                    a=policy[s]
                    sum_v+=env.P[s_,r,s,a]*(r+discount_factor*V_prev[s_])
            V[s]=sum_v
            
        error=0
        for s in states:
            error+=abs(V_prev[s]-V[s])
        
        if error<theta:
            break
            
    return V

{% endhighlight %}

Next, we create a policy improvement. The idea is, as explained above, to choose greedly.


{% highlight ruby %}
def policy_improvement(policy, value_function,env, discount_factor=1.0):
    '''policy: (deterministic) dict. A policy in the form of policy[state]=action.
       env: Enviroment.
       discount_factor: the gamma above.
       theta: error allowed.
       value_function: a dic of the form V[state]=expected gain
     '''
    states=env.states
    actions=env.actions
    rewards=env.rewards
    P=env.P
    V=value_function
    
    policy_stable=True
    for s in states:
        old_action=policy[s]
        action=old_action
        
        gain=-float('inf')
        
        for a in actions:
            expression=0
            for s_ in states:
                for r in rewards:
                    expression+=P[s_,r,s,a]*(r+discount_factor*V[s_])
            if expression>gain:
                action=a
                gain=expression
        policy[s]=action
        if action!=old_action:
            policy_stable=False
    return policy_stable
    
{% endhighlight %}

We iterate the policy improvement until reaching an optimal policy.


{% highlight ruby %}
def policy_iteration(policy, env, discount_factor=1.0, theta=0.00001,steps=False):
    
    while True:
        V=policy_eval(policy, env, discount_factor, theta)
        if steps:
            print("\n The next iteration gives: \n")
            for j in range(5):
                print('|',end="")
                for i in range(5):
                    if pol2[i,j]==0:
                        print("   UP  | ",end="")
                    elif pol2[i,j]==1:
                        print(" RIGHT | ",end="")
                    elif pol2[i,j]==2:
                        print(" DOWN  | ",end="")
                    elif pol2[i,j]==3:
                        print(" LEFT  | ",end="")
                print("\n")

        
        
        if policy_improvement(policy, V,env, discount_factor=1.0):
            break
    
    
    
    return policy
{% endhighlight %}

We are ready to try this out in one example, let's see what the optimal policy is for the environment built in the previous blog. We can use an auxiliary function to convert a non-deterministic policy to a deterministic one.


{% highlight ruby %}
def deterministic(policy,env):
        pol={}
        states=env.states
        actions=env.actions
        for s in states:
            prob=0
            action=actions[0]
            for a in actions:
                if policy[a,s]>prob:
                    action=a
                    prob=policy[a,s]
            pol[s]=action
        return pol
{% endhighlight %}

We create the environment.


{% highlight ruby %}
import sys
import numpy as np
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv
env = GridworldEnv()
{% endhighlight %}

Let's use the equiprobale policy and make it deterministic.


{% highlight ruby %}
pol1={(x,(i,j)):0.25 
      for i in range(5) for j in range(5) for x in [0,1,2,3]}
pol2=deterministic(pol1,env)
{% endhighlight %}

Let's see what we got


{% highlight ruby %}
for j in range(5):
    print('|',end="")
    for i in range(5):
        if pol2[i,j]==0:
            print("   UP  | ",end="")
        elif pol2[i,j]==1:
            print(" RIGHT | ",end="")
        elif pol2[i,j]==2:
            print(" DOWN  | ",end="")
        elif pol2[i,j]==3:
            print(" LEFT  | ",end="")
    print("\n")
{% endhighlight %}

    |   UP  |    UP  |    UP  |    UP  |    UP  | 
    
    |   UP  |    UP  |    UP  |    UP  |    UP  | 
    
    |   UP  |    UP  |    UP  |    UP  |    UP  | 
    
    |   UP  |    UP  |    UP  |    UP  |    UP  | 
    
    |   UP  |    UP  |    UP  |    UP  |    UP  | 
    


That is the current policy is to always go up. We can now use the policy iteration to find an optimal policy.


{% highlight ruby %}
policyre=policy_iteration(pol2,env, discount_factor=0.9, theta=0.00001,steps=True)
{% endhighlight %}

    
     The next iteration gives: 
    
    |   UP  |    UP  |    UP  |    UP  |    UP  | 
    
    |   UP  |    UP  |    UP  |    UP  |    UP  | 
    
    |   UP  |    UP  |    UP  |    UP  |    UP  | 
    
    |   UP  |    UP  |    UP  |    UP  |    UP  | 
    
    |   UP  |    UP  |    UP  |    UP  |    UP  | 
    
    
     The next iteration gives: 
    
    | RIGHT |    UP  |  LEFT  |    UP  |  LEFT  | 
    
    | RIGHT |    UP  |  LEFT  |    UP  |  LEFT  | 
    
    | RIGHT |    UP  |  LEFT  |    UP  |  LEFT  | 
    
    | RIGHT |    UP  |  LEFT  |    UP  |  LEFT  | 
    
    | RIGHT |    UP  |  LEFT  |    UP  |  LEFT  | 
    
    
     The next iteration gives: 
    
    | RIGHT |    UP  |  LEFT  |    UP  |  LEFT  | 
    
    |   UP  |    UP  |    UP  |  LEFT  |    UP  | 
    
    |   UP  |    UP  |    UP  |  LEFT  |    UP  | 
    
    |   UP  |    UP  |    UP  |  LEFT  |    UP  | 
    
    |   UP  |    UP  |    UP  |  LEFT  |    UP  | 
    
    
     The next iteration gives: 
    
    | RIGHT |    UP  |  LEFT  |    UP  |  LEFT  | 
    
    |   UP  |    UP  |    UP  |  LEFT  |  LEFT  | 
    
    |   UP  |    UP  |    UP  |    UP  |  LEFT  | 
    
    |   UP  |    UP  |    UP  |    UP  |  LEFT  | 
    
    |   UP  |    UP  |    UP  |    UP  |  LEFT  | 
    
    
     The next iteration gives: 
    
    | RIGHT |    UP  |  LEFT  |    UP  |  LEFT  | 
    
    |   UP  |    UP  |    UP  |  LEFT  |  LEFT  | 
    
    |   UP  |    UP  |    UP  |    UP  |    UP  | 
    
    |   UP  |    UP  |    UP  |    UP  |    UP  | 
    
    |   UP  |    UP  |    UP  |    UP  |    UP  | 
    


Which is neat! Note that after the first iteration it finds a policy that agrees with our intuition, but then it develops into an optimal one. See the book for all the other optimal ones.
