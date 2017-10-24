---
layout: post
title:  "The Markov Property"
date:   2016-11-12
category: reinforcement
---


The goal of this notebook is to understand the first class of reinforcement learning, better known as Markov chains. We start by reviewing the definitions.

**Agent:** The learner and decision maker. 

**Environment:** Everything outside the agent. What the agents interacts with.

**Rewards:** Numerical values that the agent tries to maximize over time. The reward at step $$t$$ is denoted by $$R_t$$. 

**Task:** One cycle.

**State:** The representation of the environment that the agent receives. Denoted by $$S_t$$ for $$t=0,1,2\ldots$$. The set of states is denoted by $$\mathcal{S}$$.

**Action:** What the agent does. This is choose among a set of action that depend on the state. In symbols, $$A_t \in \mathcal{A}(S_t)$$. Where, $$\mathcal{A}(S_t)$$ denotes the avaliable action under the representation of the enviroment at step $$t$$. 


[INCLUDE A GRAPHIC HERE]

It is important to note that the agent doesn't (necesarilly) knows the exact state of the environment. That's why we talk about the representation of the environment. Let's look at two cases:
- A chess game: In this case the agent, one of the payers, gets as a State the whole information about the environment.
- A poker game: The agent only gets the cards on the table and his cards as a state. It won't get the card that the other players have. He may make guesses and create a representation but, in general, it doesn't have certanty.

Now that we have a basic setup, we can begin to determine action. The idea is to assign, for a State $$A_t$$, a probability distribution over the possible actions. This is called the *policy* of the agent. We use the following notation:

$$ \pi_t(a|s) :=P(A_t=a|S_t=s) $$

to represent the probability that the agent executes the action $$a$$ at step $$t$$ when the state is $$s$$. 

# The goal

As mentioned above, the goal of the agent is to maximize the reward over time. That is, we want to maximize the quantity 

$$ R_0+R_1+R_2+\ldots $$

In general, this is a little restrictive. Instead, we try to maximize a more general function of the rewards called *the expected return* $$G_t$$ from step $$t$$ onwards, where 

$$G_t=G_t(R_{t+1},R_{t+2},\ldots)$$

The simplest example would be $$G_t=R_{t+1}+R_{t+2}+R_{t+3}+\ldots$$. But in general we may want to penalize taking too long to get rewards. We can do this with *discounting*:

$$ G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+1+k}$$

with the *discount rate* being $$0\leq \gamma \leq 1$$.



# The Markov Property

In many systems, the agent's next action only depends on the current state of the system, not all the previous states. Consider for example, chess (we only care about the current state of the board), flying a drone (we only care about the current position and velocity), etc. To capture this information we want

$$ 
\begin{multline} 
P(S_{t+1}=s', R_{t+1}=r | S_0=s_0,A_0=a_0,R_1=r_1,\ldots,S_t=s_t,A_t=a_t,R_t=r_t) \\ = P(S_{t+1}  =s', R_{t+1}=r |S_t=s_t=s,A_t=a_t=a) 
\end{multline}
$$

This is the Markov Property and we denoted the distribution function by $$p(s',r\mid s,a)$$. Under these circunstances, we will be able to predict all future states and expected rewards.

# The process

From the Markov property above we compute the expected reward as function of the state and the action as 

$$ r(s,a):= \mathbb{E}[R_{t+1} \mid s_t=s, A_t=a]=\sum_{r\in \mathcal{R}}\sum_{ s' \in \mathcal{S}}p(s',r \mid s,a), $$

the state-transiction probabilities,

$$ p(s'\mid s,a):=P(S_{t+1}=s'\mid S_t=s,A_t=a)=\sum_{r\in \mathcal{R}}p(s',r \mid s,a),$$

and the expected rewards for state-action-next-state triples

$$ r(s,a,s'):=\mathbb{E}[R_{t+1}\mid S_t=s,A_t=a,S_{t+1}=s']=\frac{\sum_{r\in\mathbb{R}}rp(s',r \mid s,a)}{p(s'\mid s,a)}.$$

# Value Functions

Given a policy $$\pi$$, we define the *value* of a state $$s$$ under the policy $$\pi$$ asthe expected return when starting in $$s$$ and following the policy $$\pi$$ from that moment on, that is:

$$ v_{\pi}(s):=\mathbb{E}_{\pi}[G_t\mid S_t=s]=\mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1} \mid S_t=s\right] $$

We also define the *action-value function* $$q_\pi$$ as the value of taking action $$a$$ in state $$s$$ under policy $$\pi$$, that is, 

$$ q_{\pi}(s,a):=\mathbb{E}_{\pi}[G_t\mid S_t=s,A_t=a]=\mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1} \mid S_t=s,A_t=a\right] $$

# The Bellman equation for $$V_\pi$$.

We now deduce a recursive equation to compute $$v_\pi$$. This is a corner stone for many techniques.

$$
\begin{align*}
v_{\pi}(s) :&= \mathbb{E}_{\pi}[G_t\mid S_t=s]\\
&= \mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1} \mid S_t=s\right]\\
& = \mathbb{E}_{\pi}\left[R_{t+1}+\gamma\sum_{k=0}^{\infty}\gamma^kR_{t+k+2} \mid S_t=s\right]\\
&= \sum_{a,s',r} \pi(a|s)p(s',r|s,a)\left[ r + \gamma \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^kR_{t+k+2} \mid S_{t+1}=s' \right] \right]\\
&= \sum_a \pi(a\mid s) \sum_{s',r} p(s',r \mid s,a)\left[ r + \gamma v_{\pi}(s')\right].
\end{align*}
$$

The *optimal state-value function*, denoted by $$v_*$$, and defined as

$$v_*:= \max_{\pi} v_{\pi}(s), $$

where $$\pi\geq \pi'$$ if and only if $$v_{\pi}(s)\geq v_{\pi'}(s)$$ for all $$s$$.

Similarly, the *optimal action-value function* is 

$$q_*(s,a):=\max_{\pi} q_{\pi}(s,a).$$

Note that 

$$q_*(s,a) = \mathbb{E}[R_{t+1}+\gamma v_*(S_{t+1}) \mid S_t=s,A_t=a].$$

More importantly, the Bellman equation in this case has a better descrition. (Recall, that a policy determines the probability of choosing the action $$a$$ under the state $$s$$, hence if we have selected "the best" policy then this probability centers over one action and we get 

$$
\begin{align*}
v_*(s)&=\max_{a\in \mathcal{A}(s)} q_{\pi_*}(s,a)\\
&=\max_a \mathbb{E}_{\pi_*} \left[ G_t \mid S_t=s, A_t=a \right]\\
&= \max_a \mathbb{E}_{\pi_*} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t=s, A_t =a \right]\\
&= \max_a \mathbb{E}_{\pi_*} \left[ R_{t+1}+ \gamma v_*(S_{t+1}) \mid S_t=s, A_t =a \right]\\
&=\max_a \sum_{s',r} p(s',r \mid s,a)[r+\gamma v_*(s')]
\end{align*}
$$

Similarly,

$$
\begin{align*}
q_*(s,a)&=\mathbb{E}[R_{t+1}+\gamma \max_{a'}q_*(S_{t+1},a')\mid S_t=a,A_t=a ] \\
& = \sum_{s',r} p(s',r \mid s,a)\left[ r + \gamma \max_{a'} q_*(s',a')\right].
\end{align*}
$$

These are functional equations on $$v_*$$ and $$q_*$$, our goal is to find their solutions. Which in the case of finite Markov Decission Process (MDP), it is unique solution. 


{% highlight ruby %}

{% endhighlight %}
