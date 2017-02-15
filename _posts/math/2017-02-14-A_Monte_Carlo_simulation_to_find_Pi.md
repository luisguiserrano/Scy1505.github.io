---
layout: post
title:  "A Monte-Carlo simulation to find Pi"
date:   2017-02-14
category: math
---


In this post, we explain what a Monte-Carlo simulation is and use the idea to approximate the value of $$\pi$$. This notes are intended for MATH 4991.

# The Monte-Carlo Simulation

Before getting into the simulations, let's recall some basic definitions:

- The *probability of an event $$A$$* is understood as either the degree of believe that the event $$A$$ will happen, or as the ratio of times that event will happen against all events. We denote this by $$p(A)$$. 

**Example:** The proability of getting 3 from a dice is 1/6.

In simple terms, a Monte-Carlo Simulation is a technique takes advantage of the second interpretation of probability. It is use to  assest risk of an event given a probabilistic distribution. 

Let's be more precise. Suppose that we want to understand the likeness for an event $$A$$ to take place if we know that an event $$B$$ has already happened. This is denoted by $$p(A\mid_B)$$, by the second definition we just need to let event $$A$$ to happen many times and count how many of those times we obtained the event $$B$$, that is

$$ P(A\mid_B)=\lim_{\text{repetitions}\to \infty} \frac{\text{number of times A happens}}{ \text{number of repetitions of event B}}.$$

That is for computing the probability of getting $$3$$ from a dice, we just throw the dice many times and count how many of those  we got $$3$$. You can try this yourself and convince after 6000 times  about 1000 of those times you got a 3. **Wait!** Who has the time to throw a dice 6000 times? Luckily, computers can simulate the throw of a dice fairly easy.

## The random package

Most languages come with a package that creates random numbers, in Python we use the package random. We import it with the following line.


{% highlight ruby %}
import random
{% endhighlight %}

For our purposes we need 2 methods from this package. The method random(), and the method randint(). You can find the documentation [here](https://docs.python.org/2/library/random.html).

random.random() gives a random real number between 0 and 1.


{% highlight ruby %}
random.random()
{% endhighlight %}




    0.6763334359339186



and random.randint(a,b) gives a random integer between a and b, both included.


{% highlight ruby %}
random.randint(0,10)
{% endhighlight %}




    7



So the throw of dice can be modeled by random.randint(1,6). Let's throw the dice 3 times.


{% highlight ruby %}
random.randint(1,6)
{% endhighlight %}




    3




{% highlight ruby %}
random.randint(1,6)
{% endhighlight %}




    2




{% highlight ruby %}
random.randint(1,6)
{% endhighlight %}




    3



We can use the for loop to throw the dice as many times as we want. So let's throw it 6000 times and count how many times we get 3.


{% highlight ruby %}
# We haven't throw the dice so we got not 3 yet
got_three=0
for i in range(6000):
    
    dice_gives=random.randint(1,6)
    
    #If the dice gives a 3 then added to the count
    if dice_gives==3:
        got_three=got_three+1
        
#Finally, we compute the ratio
probability=got_three/6000

#And print the number of 3 and the probability
print("The number of 3's is "+str(got_three))
print("Which gives a probability of "+str(probability))
{% endhighlight %}

    The number of 3's is 978
    Which gives a probability of 0.163


## The value of $$\pi$$

Consider the following problem. You randomly throw darts to a 2x2 board. Inside the board, there is a circle of radius 1. 

<center>
<img src="{{ '/assets/img/A_Monte_Carlo_simulation_to_find_Pi_files/circle2.png' | prepend: site.baseurl }}" alt=""> 
</center>

The question is then, what is the probability of hitting the circle? 

**The math solution:** Let $$A$$ be the event of the dart hitting inside the circle, then 

$$p(A)=\frac{\text{Area circle}}{\text{Area square}}=\frac{\pi}{4}$$

** The modeling solution: ** We just throw many darts! We randomly select points $$p=(x,y)$$ with $$0\leq x,y \leq 2$$. Note that if the distance of $$p$$ to the point $$(1,1)$$ is less than 1, then the point $$p$$ is inside the circle. That is we need to check if

$$(x-1)^2+(y-1)^2<1$$

to know if $$p$$ is inside the circle.

We present five different ways to do this.

### Version 1

This is, probably, the most basic approach.


{% highlight ruby %}
#Import the random method from the random library.
from random import random


#Initialize the counter for the darts.
inside=0

#Throw 300 darts
for i in range(300):
    
    # Create the dart, remember random() gives a number between 0 and 1, 
    # so 2*random() gives a random number between 0 and 2.
    x=2*random()
    y=2*random()
    
    #Check if the dart is inside the circle, if so increase the counter by 1.
    if (x-1)**2+(y-1)**2<1:
            inside=inside+1
            
#divides
probability=inside/300
print("The probaility we get is "+str(probability))

#Note that 4 times the probability should be pi, let's check this.
pi_approx=4*probability
print("An approximation for Pi is "+str(pi_approx))

{% endhighlight %}

    The probaility we get is 0.7566666666666667
    An approximation for pi is 3.026666666666667


**Pros:** 

- Simple model. 
- Well documented.

**Cons:** 

- If want to throw more darts we need to change two lines of code.
- We need to tune by hand if we want a good approximation.
- No graphical description.

### Version 2

This is a show off version. This is called a one-liner.


{% highlight ruby %}
import random as rd
sum([1 for i in range(3000) if (2*rd.random()-1)**2+(2*rd.random()-1)**2<1])/3000*4
{% endhighlight %}




    3.1373333333333333



**Pros:** 

- Clever. 
- Compact.

**Cons:** 

- If want to throw more darts we need to change the code in two places.
- No documentation, it doesn't tell if it is approximating Pi or finding the probability.
- Can not be tune to get preccision.
- No graphical description.

### Version 3

This model is more organized, it keeps the information so it can be analized later if needed and creates a graph.


{% highlight ruby %}
import random as rd
from matplotlib import pyplot as plt

# Create placeholders to keep the coordinates of the points inside and outside of the circle.
x_coord_in=[]
y_coord_in=[]
x_coord_out=[]
y_coord_out=[]


# Iterate over number of throws, 100 in this case.
for i in range(100):
    
    # Create the dart, remember random() gives a number between 0 and 1, 
    # so 2*random() gives a random number between 0 and 2.
    x=2*rd.random()
    y=2*rd.random()
    
    # Check if the dart is inside the circle.
    # If so adds its coordinates to the lists x_coord_in y_coord_in.
    # If NOT adds its coordinates to the lists x_coord_out y_coord_out.
    if (x-1)**2+(y-1)**2<1:
        x_coord_in.append(x)
        y_coord_in.append(y)
    else:
        x_coord_out.append(x)
        y_coord_out.append(y)

        
# Compute the probability
probability = len(x_coord_in)/100


# Compute the approximation of pi
pi_approx=4*probability


# The following code makes plot ----------------------------------------

# Creates a canvas to draw.
fig = plt.figure(figsize=(5, 5), dpi=100)

# Add the points that hit the circle to the canvas and marks them with x. 
plt.plot(x_coord_in, y_coord_in, 'x')

# Add the points that hit the circle to the canvas and marks them with point.
plt.plot(x_coord_out, y_coord_out, '.')

# Create a circle of radius 1 and center (1,1) to the canvas, paints it red.
circle = plt.Circle((1, 1), 1, color='r')

# Add the circle to the canvas
plt.gca().add_artist(circle)

# Add a tittle 
plt.title('100 tries')

# Plot the axis
plt.axis([0, 2, 0, 2])

# Show the canvas
plt.show()

# End of plot ---------------------------------------------------------


# display the probabily
print("The probaility we get is "+str(probability))


# Display the approximation of pi
print("An approximation for Pi is "+str(pi_approx))


{% endhighlight %}


<center>
<img src="{{ '/assets/img/A_Monte_Carlo_simulation_to_find_Pi_files/A_Monte_Carlo_simulation_to_find_Pi_30_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


    The probaility we get is 0.76
    An approximation for Pi is 3.04


**Pros:** 

- Clear documentation. 
- Graphical description.


**Cons:** 

- If want to throw more darts we need to change the code in several places.
- Can not be tune to get preccision.
- Maybe too verbose?

### Version 4

One way to deal with the cons we had found before is to make the model a function.


{% highlight ruby %}
def probability(iterations=100, display_graph=False, verbose=True):
    ''' Aproximates the probability of hitting the circle using the Monte-Carlo method
    args:
        iteration (int): Number of iterations.
        display_graph (bol): Boolean determing if showing graph.
        verbose (bol): Boolean determing if showing the info.
    returns: 
        the probability (float)
    '''
    
    import random as rd
    from matplotlib import pyplot as plt
    
    # Create placeholders to keep the coordinates of the points inside and outside of the circle.
    x_coord_in=[]
    y_coord_in=[]
    x_coord_out=[]
    y_coord_out=[]
    
    
    # Iterate over iteration number of throws.
    for i in range(iterations):
        
        # Create the dart.
        x=2*rd.random()
        y=2*rd.random()
        
        # Check if the dart is inside the circle.
        # If so adds its coordinates to the lists x_coord_in y_coord_in.
        # If NOT adds its coordinates to the lists x_coord_out y_coord_out.
        if (x-1)**2+(y-1)**2<1:
            x_coord_in.append(x)
            y_coord_in.append(y)
        else:
            x_coord_out.append(x)
            y_coord_out.append(y)

    #Compute the probability
    probability = len(x_coord_in)/iterations
    
    if display_graph:
        
        # make plot    
        fig = plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(x_coord_in, y_coord_in, 'x')
        plt.plot(x_coord_out, y_coord_out, '.')
        circle = plt.Circle((1, 1), 1, color='r')
        plt.gca().add_artist(circle)
        plt.title('%d iterations'%iterations)
        plt.axis([0, 2, 0, 2])
        plt.show()
        
    #display the probability
    if verbose:
        print("The probability of hitting the circle with %d iterations is %.3f "%(iterations, probability))
    
    #return the value
    return probability

def approx_pi(error=0.001,step=1000):
    
    current_error=float('inf')
    
    iterations=1000
    
    approx_A = 4*probability(iterations, verbose=False)
    approx_B = 4*probability(iterations+step, verbose=False) 
    
    current_error = abs(approx_A-approx_B)
    counter=0
    while error<current_error:
        
        approx_A = approx_B
        approx_B = 4*probability(iterations+step, verbose=False) 
        
        iterations = iterations+step
        
        current_error = abs(approx_A-approx_B)
        
        counter+=1
        
        if counter>100000:
            break
    
    print("With an accuracy of %f the value of pi is %f and it took %d iterations"%(error, approx_B,iterations))
{% endhighlight %}

With these functions we can compute any number of examples that we want.


{% highlight ruby %}
probability(iterations=10, display_graph=True)
probability(iterations=100, display_graph=True)
probability(iterations=1000, display_graph=True)
probability(iterations=10000, display_graph=True)
{% endhighlight %}


<center>
<img src="{{ '/assets/img/A_Monte_Carlo_simulation_to_find_Pi_files/A_Monte_Carlo_simulation_to_find_Pi_36_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


    The probability of hitting the circle with 10 iterations is 0.800 



<center>
<img src="{{ '/assets/img/A_Monte_Carlo_simulation_to_find_Pi_files/A_Monte_Carlo_simulation_to_find_Pi_36_2.png' | prepend: site.baseurl }}" alt=""> 
</center>


    The probability of hitting the circle with 100 iterations is 0.730 



<center>
<img src="{{ '/assets/img/A_Monte_Carlo_simulation_to_find_Pi_files/A_Monte_Carlo_simulation_to_find_Pi_36_4.png' | prepend: site.baseurl }}" alt=""> 
</center>


    The probability of hitting the circle with 1000 iterations is 0.788 



<center>
<img src="{{ '/assets/img/A_Monte_Carlo_simulation_to_find_Pi_files/A_Monte_Carlo_simulation_to_find_Pi_36_6.png' | prepend: site.baseurl }}" alt=""> 
</center>


    The probability of hitting the circle with 10000 iterations is 0.781 

    0.7809



And we can find an approximation of $$\pi$$. 

**Warning:** Be careful with the choices for error and step, it may take a long time to compute.


{% highlight ruby %}
approx_pi(error=0.001,step=1000)
approx_pi(error=0.0001,step=10000)
approx_pi(error=0.0001,step=100000)
{% endhighlight %}

    With an accuracy of 0.001000 the value of pi is 3.143000 and it took 12000 iterations
    With an accuracy of 0.000100 the value of pi is 3.143786 and it took 131000 iterations
    With an accuracy of 0.000100 the value of pi is 3.141735 and it took 3401000 iterations


Note that the values that this gives may still be outside the error value of $$\pi$$ since we are not requiring many repetations to make sure the error is correct with respect .

**Exercise:** Modify the code above to have more certantity on the value of $$\pi$$.

**Pros:** 

- Clear documentation. 
- Graphical description.
- We can choose the number of throws.
- We can specify the preccision for pi
- Modular structure so it is easy to modify and adapt to other problems.


### Version 5

This version creates an animation so we can see what's going on in real time. This is more advanced and requires a little bit more background.


{% highlight ruby %}
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np

# New figure with white background
fig = plt.figure(figsize=(4,4), facecolor='white')

# New axis over the whole figure, no frame and a 1:1 aspect ratio
ax = fig.add_axes([0,0,1,1], frameon=False, aspect=1)

# Place holder for darts position in and out
P_in=np.empty((1,2))
P_out=np.empty((1,2))


# Scatter plots for in and out
scat_in = ax.scatter([], [], s=60,lw = 1, marker='x',color='b')
scat_out = ax.scatter([], [], s=30,lw = 1, marker='o',color='g')

# Ensure limits are [0,2] and remove ticks
ax.set_xlim(-0.4,2.3), ax.set_xticks([])
ax.set_ylim(-0.4,2.3), ax.set_yticks([])

# Text to be displayed
iteration_text = ax.text(1-0.3, 0.02, 'Iteration: 0 ', transform=ax.transAxes)
probability_text = ax.text(0.02, 0.02, 'probability: 0.000', transform=ax.transAxes)
pi_approx_text= ax.text(0.02, 0.07, '$$\pi$$ approx: 0.000', transform=ax.transAxes)


# Adds target to drawing.
circle = plt.Circle((1, 1), 1, color='r',alpha=0.8,zorder=0)
ax.add_artist(circle)

lines =plt.plot([0,2,2,0,0],[0,0,2,2,0],color='black')


def update(frame):
    
    #Calls the placeholders
    global P_in, P_out

    # Create a dart
    new = np.random.uniform(0,2,(1,2))
        
    #Check if the dart is inside the circle
    if np.linalg.norm(new-np.array([[1,1]]))<1:
        P_in = np.concatenate((P_in,new))
    else:
        P_out=np.concatenate((P_out,new))
    
    # Place the points in the graph
    scat_in.set_offsets(P_in)
    scat_out.set_offsets(P_out)
    
    #Shows the current iteration
    iteration_text.set_text('Iteration: %d '%frame)
    
    #Every 50 iterations if shows the probability and approx of Pi
    if frame%50==0:
        probability=len(P_in)/(len(P_in)+len(P_out))
        pi_approx=4*probability
    
        probability_text.set_text('probability: %.3f'%probability )
        pi_approx_text.set_text('$$\pi$$ approx: %.3f'%pi_approx)
    
    # Return the modified object
    return scat_in, scat_out,iteration_text, probability_text,pi_approx_text.set_text

ani = animation.FuncAnimation(fig, update, interval=100, frames=1001)

#The following line is optional, it saves the video animation as a gif file.
#ani.save('darts.gif', writer='imagemagick', fps=30, dpi=40)


HTML(ani.to_html5_video())

{% endhighlight %}


<center>
<img src="{{ '/assets/img/A_Monte_Carlo_simulation_to_find_Pi_files/darts.gif' | prepend: site.baseurl }}" alt=""> 
</center>





You can see some other example [here](https://pythonprogramming.net/monte-carlo-simulator-python/), you can also find a nice introduction to plotting in Python [here](https://www.labri.fr/perso/nrougier/teaching/matplotlib/#text).
