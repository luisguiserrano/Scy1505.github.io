---
layout: post
title:  "A data manual for Baobei - The Titanic Data Example"
date:   2016-10-11
published: true
---



Hi Baobei, 

The puppy scientific learning community is happy to have you on board with this course in the basic techniques of Machine Learning. So, Welcome!

## The Titanic Data

The RMS Titanic was a ship that sunk in 1912. The failure to have proper safety procedures and equipement led to a huge loss of life. Our goal is to find the features of a person that is more likely to survived. Before talking about the data set let's load the packages we will need:
- numpy: Allows to deal with scientific computations.
- pandas: Helps dealing with datasets, series, and data related stuff.
- sklearn: Short for scientific kit for learning, contains several methods for machine learning. Quite useful!
- matplotlib: Package for plotting graphs and visualization of data.


{% highlight ruby %}
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
%matplotlib inline
{% endhighlight %}

The last line here %matplotlib inline allows for the matplotlib graphs to appear in line. We now load the data.


{% highlight ruby %}
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
{% endhighlight %}

The data is load from a csv (comma separated value) file, a database file extension compatible with most systems. The data is saved in an object from the Pandas package call DataFrame.


{% highlight ruby %}
type(train)
{% endhighlight %}




    pandas.core.frame.DataFrame



Also note that the data came in two files. It is common in machine learning to divide the data into two parts, one for training and one for testing. In this case it was divided by 2/3 for training and 1/3 for testing. (This is a little more uncommon, usually some percentage of the data is use for testing, I think the reason of the choice in this case is because the data set has a small size). I also want to mention that in general is not a good idea to divide the data arbitrarly, there are several tools in the sklearn library to split the data such that the test set is as representative of the whole data as possible (More abbout this in other notebook).

Now that we got the data, we would like to know how it looks like. We can use the method [DataFrame.describe()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html)  to do this. 


{% highlight ruby %}
train.describe()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight ruby %}
test.describe()
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>332.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>417.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1100.500000</td>
      <td>2.265550</td>
      <td>30.272590</td>
      <td>0.447368</td>
      <td>0.392344</td>
      <td>35.627188</td>
    </tr>
    <tr>
      <th>std</th>
      <td>120.810458</td>
      <td>0.841838</td>
      <td>14.181209</td>
      <td>0.896760</td>
      <td>0.981429</td>
      <td>55.907576</td>
    </tr>
    <tr>
      <th>min</th>
      <td>892.000000</td>
      <td>1.000000</td>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>996.250000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.895800</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1100.500000</td>
      <td>3.000000</td>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1204.750000</td>
      <td>3.000000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1309.000000</td>
      <td>3.000000</td>
      <td>76.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



This gives the info for both DataFrames, note that there are 891 passangers in the train dataset and 418 in the test dataset. Furthermore, we get the basic info like the mean, standard deviation, etc. 

But how does the data looks like? We can check the whole data by typing it. That is just typing train (Try it!). But this would look clutter, so instead we can get a sample by using the [method DataFrame.head()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html) or [method DataFrame.tail()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.tail.html)


{% highlight ruby %}
train.head(n=3)
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



and for the test DataFrame


{% highlight ruby %}
test.tail(n=3)
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>415</th>
      <td>1307</td>
      <td>3</td>
      <td>Saether, Mr. Simon Sivertsen</td>
      <td>male</td>
      <td>38.5</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/O.Q. 3101262</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>416</th>
      <td>1308</td>
      <td>3</td>
      <td>Ware, Mr. Frederick</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>359309</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1309</td>
      <td>3</td>
      <td>Peter, Master. Michael J</td>
      <td>male</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>2668</td>
      <td>22.3583</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



The first thing we notice is that the feature "Survived' appears in the train DataFrame, but it does not appear in the test DataSet, the reason is simple, this is what we want to predict. Before using machine learning techniques, we should understand how the data looks like and create ourselves some "common sense predictions". To look at the data we use the matplotlib library. 

### Making the graphs look nice 

The matplotlib library is very versitile and will let us create beautiful graphs. Furthermore, the panda library uses the matplotlib and increases the type of graphs we can create. We first set some global parameters for the matplotlib, we can use the method [matplotlib.rc()](http://matplotlib.org/users/customizing.html) to do this.


{% highlight ruby %}
plt.rc('font', size=12, family='fantasy')
{% endhighlight %}

### A first look at the data

From the data description above we note that there two numeric features: Age and Fare, and some categorical ones: Pclass, Sex, Embarked, etc. Let's study the numerical ones first. We will create density distributions for the Age and Fare using the KDE plot option for [DataFrame.plot()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html#pandas.DataFrame.plot).


{% highlight ruby %}
fig = plt.figure(figsize=(18, 4)); # We create our canvas.
alpha = 0.7; # How transparent curves will be.

# Subdivides the canvas in one row and two columns and says we are plotting at the one with coordinates (0,0)
ax0 = plt.subplot2grid((1,2), (0,0)) 


# Obtains the feature Age from the data set, then it plots this as a density function using the kde type for DataFrame.plot()
train.Age.plot(kind='kde', color='#FA2379', label='train', alpha=alpha)
test.Age.plot(kind='kde', label='test', alpha=alpha)

# Places the labes to the axis an graph
ax0.set_xlabel('Age')
ax0.set_title("Density probability for Age" )

#place the legend in the best possible position
plt.legend(loc='best');

# Subdivides the canvas in one row and two columns and says we are plotting at the one with coordinates (0,1)
ax1 = plt.subplot2grid((1,2), (0,1))

# Obtains the feature Fare from the data set, then it plots this as a density function using the kde type for DataFrame.plot()
train.Fare.plot(kind='kde', color='#FA2379', label='train', alpha=alpha)
test.Fare.plot(kind='kde', label='test', alpha=alpha)

# Places the labes to the axis an graph
ax1.set_xlabel('Fare')
ax1.set_title("Density probability for Fare" )

#place the legend in the best possible position
plt.legend(loc='best');

{% endhighlight %}

<img src="{{ '/img/Titanic_files/Titanic_24_1.png' | prepend: site.baseurl }}" alt=""> 



Next, we use bar graphs to show the Proportions of passengers in the categories Pclass, Sex, and Embarked. The idea is always the same, first we use DataFrame.Feature to create the data series associated to the feature, then [Series.value_counts()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html) to create the data series associated to the relative frequencies of the unique values (note that this is achieved by the parameter normalize). Finally we use the plot method with the parameter kind set to 'bar' to get the bars graph.


{% highlight ruby %}
fig = plt.figure(figsize=(18, 4)); # We create our canvas.
alpha = 0.7; # How transparent curves will be.

# Subdivides the canvas in one row and three columns and says we are plotting at the one with coordinates (0,0)
ax0 = plt.subplot2grid((1,3), (0,0)) 

# Obtains the feature Pclass from the data set, then plots this as bar graphs using the bar type for DataFrame.plot()
train.Pclass.value_counts(normalize=True).sort_index().plot(kind='bar', color='#FA2379', label='train', alpha=alpha)
test.Pclass.value_counts(normalize=True).sort_index().plot(kind='bar', label='test', alpha=alpha)

# Places the labes to the axis an graph
ax0.set_xlabel('Passanger Classs')
ax0.set_title('Passanger ratio by Class')

#place the legend in the best possible position
ax0.legend(loc='best')

# Subdivides the canvas in one row and three columns and says we are plotting at the one with coordinates (0,0)
ax1 = plt.subplot2grid((1,3), (0,1)) 

# Obtains the feature Sex from the data set, then plots this as bar graphs using the bar type for DataFrame.plot()
train.Sex.value_counts(normalize=True).sort_index().plot(kind='bar', color='#FA2379', label='train', alpha=alpha)
test.Sex.value_counts(normalize=True).sort_index().plot(kind='bar', label='test', alpha=alpha)

# Places the labes to the axis an graph
ax1.set_xlabel('Sex')
ax1.set_title('Passanger ratio by Sex')

#place the legend in the best possible position
ax1.legend(loc='best')

# Subdivides the canvas in one row and three columns and says we are plotting at the one with coordinates (0,0)
ax2 = plt.subplot2grid((1,3), (0,2)) 

# Obtains the feature Sex from the data set, then plots this as bar graphs using the bar type for DataFrame.plot()
train.Embarked.value_counts(normalize=True).sort_index().plot(kind='bar', color='#FA2379', label='train', alpha=alpha)
test.Embarked.value_counts(normalize=True).sort_index().plot(kind='bar', label='test', alpha=alpha)

# Places the labes to the axis an graph
ax2.set_xlabel('Embarked')
ax2.set_title('Passanger ratio by Embarked')

#place the legend in the best possible position
ax2.legend(loc='best');


{% endhighlight %}

<img src="{{ '/img/Titanic_files/Titanic_26_0.png' | prepend: site.baseurl }}" alt=""> 



A note about the feature Embarked. It means Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).

Before comparing this different features with the likeness of survival, let's see how many people survived. 


{% highlight ruby %}
train.Survived.value_counts()
{% endhighlight %}




    0    549
    1    342
    Name: Survived, dtype: int64



That is, only 342 people survived in our training set, that is about 38%. It is reasonable to expect the same is true for our testing set, as long as the choice of the testing data was homogeneous with respect to the whole data set.

### Are younger people more likely to survive?

We can create a Data Series of the people who survived and look at their ages. We could do the same for people who didn't survived. 


{% highlight ruby %}
fig = plt.figure(figsize=(18,5))
alpha=0.7

train[train.Survived==0].Age.plot(kind='kde',color='#FA2379',label='Not Survived',alpha=alpha)
train[train.Survived==1].Age.plot(kind='kde',label='Survived',alpha=alpha)
plt.legend(loc='best');

plt.xlabel('Age')
plt.title("What age group is more likely to survive?" );
{% endhighlight %}

<img src="{{ '/img/Titanic_files/Titanic_33_0.png' | prepend: site.baseurl }}" alt=""> 



We can draw a couple of quick conclutions. First, there's a small increase in the chances of survival for people between ~15  and 30 years old. Second, the chances of survival decreasse if you are less than 15. But the chances do not change if older than 30.

### Are men or women more likely to survive?

We can proceed as before, but select the sex feature instead.


{% highlight ruby %}
fig = plt.figure(figsize=(18,5))
alpha=0.7

train[train.Survived==1].Sex.value_counts(normalize=True).plot(kind='bar',label='Survived',alpha=alpha)
plt.legend(loc='best');

plt.xlabel('Age')
plt.title("What sex is more likely to survive?" );
{% endhighlight %}

<img src="{{ '/img/Titanic_files/Titanic_37_0.png' | prepend: site.baseurl }}" alt=""> 



That is about two thirds of the survivors are women and one third is men.


{% highlight ruby %}
fig = plt.figure(figsize=(18,5))
alpha=0.7

train[train.Survived==0].Sex.value_counts(normalize=True).plot(kind='bar',color='#FA2379',label='Not Survived',alpha=alpha)

plt.legend(loc='best');

plt.xlabel('Age')
plt.title("What sex is more likely to no  survive?" );
{% endhighlight %}

<img src="{{ '/img/Titanic_files/Titanic_39_0.png' | prepend: site.baseurl }}" alt=""> 



And less than one fifth of the ones that didn't survived is female. We can conclude that there are better chances of surviving for female persons.  Let's try to be more precise and divide the population into four subsets, FemSur, FemNoSur, MalSur, and MalNoSur. We create a new feature to keep this information, we will need an auxiliary function.


{% highlight ruby %}
def AuxSexSur(C):
    if C['Sex']=='female' and C['Survived']==1:
        return 'Female Survivor'
    elif C['Sex']=='female' and C['Survived']==0:
        return 'Female No Survivor'
    elif C['Sex']=='male' and C['Survived']==1:
        return 'Male Survivor'
    elif C['Sex']=='male' and C['Survived']==0:
        return 'Male No Survivor'
{% endhighlight %}

And we build the new feature in a copy of our DataFrame train.


{% highlight ruby %}
train2=train.copy()
train2['SexSur']=train2.apply(AuxSexSur,axis=1)
{% endhighlight %}

We can now plot this data, we use a pie chart for this.


{% highlight ruby %}
fig = plt.figure(figsize=(4,4))

train2.SexSur.value_counts(normalize=True).plot(kind='pie',label='Sex and Survival')

plt.title("What sex is more likely to no  survive?" );
{% endhighlight %}

<img src="{{ '/img/Titanic_files/Titanic_45_0.png' | prepend: site.baseurl }}" alt=""> 


This give us the outcome that if female the chances are about 2/3 of surviving meanwhile if male the chances are about 1/5.

### Survival with respect to class

This is another likely factor into surival, so it is worth taking a look at it. Let's plot what were the survival ratios in each of the classes.


{% highlight ruby %}
fig=plt.figure(figsize=(18,5))
alpha=0.6

ax0=plt.subplot2grid((1,3),(0,0))
train[train.Pclass==1].Survived.value_counts(normalize=True).sort_index().plot(kind='bar')
ax0.set_title('First Class')
ax0.set_ylabel('Percentage')
ax0.set_xlabel('1: Survived\n 0: Did not survive')

ax0=plt.subplot2grid((1,3),(0,1))
train[train.Pclass==2].Survived.value_counts(normalize=True).sort_index().plot(kind='bar',color='#FA2379')
ax0.set_title('Second Class')
ax0.set_ylabel('Percentage')
ax0.set_xlabel('1: Survived\n 0: Did not survive')

ax0=plt.subplot2grid((1,3),(0,2))
train[train.Pclass==3].Survived.value_counts(normalize=True).sort_index().plot(kind='bar',color='#78AB46')
ax0.set_title('Third Class')
ax0.set_ylabel('Percentage')
ax0.set_xlabel('1: Survived\n 0: Did not survive');

{% endhighlight %}

<img src="{{ '/img/Titanic_files/Titanic_49_0.png' | prepend: site.baseurl }}" alt=""> 



This clearly shows that people in upper classes had a highest chance of survival. But how do males and females survivale compares in each classs? We create DataFrames with only males and one with other males to find this out.


{% highlight ruby %}
train_male = train[train.Sex=='male']
train_female=train[train.Sex=='female']
{% endhighlight %}

We can now place all the information together.


{% highlight ruby %}
fig=plt.figure(figsize=(18,10))
alpha=0.6

ax1=plt.subplot2grid((2,3),(0,0))
train_female[train_female.Pclass==1].Survived.value_counts(normalize=True).sort_index().plot(kind='bar',color='#FA2379');
ax1.set_title('First Class Female')
ax1.set_ylabel('Percentage')
ax1.set_xlabel('1: Survived\n 0: Did not survive')

ax1=plt.subplot2grid((2,3),(0,1))
train_female[train_female.Pclass==2].Survived.value_counts(normalize=True).sort_index().plot(kind='bar',color='#FA2379');
ax1.set_title('Second Class Female')
ax1.set_ylabel('Percentage')
ax1.set_xlabel('1: Survived\n 0: Did not survive')

ax2=plt.subplot2grid((2,3),(0,2))
train_female[train_female.Pclass==3].Survived.value_counts(normalize=True).sort_index().plot(kind='bar',color='#78AB46');
ax2.set_title('Third Class Female')
ax2.set_ylabel('Percentage')
ax2.set_xlabel('1: Survived\n 0: Did not survive')

ax3=plt.subplot2grid((2,3),(1,0))
train_male[train_male.Pclass==1].Survived.value_counts(normalize=True).sort_index().plot(kind='bar');
ax3.set_title('First Class Male')
ax3.set_ylabel('Percentage')
ax3.set_xlabel('1: Survived\n 0: Did not survive')

ax4=plt.subplot2grid((2,3),(1,1))
train_male[train_male.Pclass==2].Survived.value_counts(normalize=True).sort_index().plot(kind='bar',color='#FA2379');
ax4.set_title('Second Class Male')
ax4.set_ylabel('Percentage')
ax4.set_xlabel('1: Survived\n 0: Did not survive')

ax5=plt.subplot2grid((2,3),(1,2))
train_male[train_male.Pclass==3].Survived.value_counts(normalize=True).sort_index().plot(kind='bar',color='#78AB46');
ax5.set_title('Third Class Male')
ax5.set_ylabel('Percentage')
ax5.set_xlabel('1: Survived\n 0: Did not survive');

plt.tight_layout() # This command allows the graph to loook nice, try to run it without it to find what happens.
{% endhighlight %}

<img src="{{ '/img/Titanic_files/Titanic_53_0.png' | prepend: site.baseurl }}" alt=""> 



This shows that first class females were by far the most likely group to survived. At heart this process we carried out is similar to a decision tree, we will come back to this later. We could keep doing this kinds of analysis and I encourage you to try your own. But, for us the next step is to clean the Data. 

## Cleaning the Data

Data usually comes messy, luckily for us, the data is quite clean. The one thing that we can see in this data is that there is plenty of missing Data. We can see how much by using the DataFrame method [DataFrame.isnull()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.isnull.html). This method returns a DataFrame object whose values are True if the data is missing or Null, and it returns False otherwise. We also use the method sum(), this counts the values associated with the feature.    


{% highlight ruby %}
train.isnull().sum()
{% endhighlight %}




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64



Note, that there are 177 ages missing, 687 missing Cabin features, and 2 for the feature embarked. We should also look at the testing data.


{% highlight ruby %}
test.isnull().sum()
{% endhighlight %}




    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64



For the testing set the missing info corresponds to age, Fare, and Cabin. We have two options for dealing with the missing data. We could get rid of this entries or we could try to approximate their values. The first option is not a good idea since our data set is already small, so our only option is approximate this data.

### Port of Embark

In [here](https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic/notebook)  Doctor Megan Risdal notes that there's a relation between the port of embarking, the Passanger class and the Fare. We use this to find what are the reasonable values for the missing port of embarking. Let's first check whose info are we missing.


{% highlight ruby %}
train[train.Embarked.isnull()]
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>62</td>
      <td>1</td>
      <td>1</td>
      <td>Icard, Miss. Amelie</td>
      <td>female</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>80.0</td>
      <td>B28</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>829</th>
      <td>830</td>
      <td>1</td>
      <td>1</td>
      <td>Stone, Mrs. George Nelson (Martha Evelyn)</td>
      <td>female</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>80.0</td>
      <td>B28</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Now we can use a boxplot graph to find the most likely value.


{% highlight ruby %}
fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(111)
ax = train.boxplot(column='Fare', by=['Embarked','Pclass'], ax=ax)
plt.axhline(y=80, color='green')
ax.set_title('', y=1.1);
{% endhighlight %}

<img src="{{ '/img/Titanic_files/Titanic_65_0.png' | prepend: site.baseurl }}" alt=""> 


That is for these passengers, the most likely scenario is that their port of embarking was 'c', which corresponds to Cherbourg. But is this correct? this seems reasonable just by reading from the Data, but it is usually better to look for "real" sources to complete the data, for example in [Encyclopedia Titanica](https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html) we can find info about the passengers. For this case we have that Amelie Icard was Mrs. Stone maid and they boarded in Southampton, that is the port of embarking should be 'S'.


{% highlight ruby %}
train.set_value(train.Embarked.isnull(), 'Embarked', 'S');
{% endhighlight %}

### The Cabin

For the missing cabins, our only option is to declare them unknown. As we want to do this in both dataframes, we just assign a label representing unknown for each DataFrame.


{% highlight ruby %}
train.set_value(train.Cabin.isnull(), 'Cabin', 'Unk')
test.set_value(test.Cabin.isnull(),'Cabin','Unk');
{% endhighlight %}

### The Fare

We now deal with the missing Fare value in the test set. Clearly, the Fare should depend on the class and the embarking port. Let's find what does the person with the missing person looks like.


{% highlight ruby %}
test[test.Fare.isnull()==True]
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>152</th>
      <td>1044</td>
      <td>3</td>
      <td>Storey, Mr. Thomas</td>
      <td>male</td>
      <td>60.5</td>
      <td>0</td>
      <td>0</td>
      <td>3701</td>
      <td>NaN</td>
      <td>Unk</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



We have several options for the Fare. The mean, the most common value, or the fifty percentile. Let's find these values for passangers in third class embarked at Southampton (Embarked='S'). First the most common value:


{% highlight ruby %}
pd.concat([train[(train.Pclass==3)&(train.Embarked=='S')].Fare,test[(test.Pclass==3)& (test.Embarked=='S')].Fare]).value_counts().head()
{% endhighlight %}




    8.0500    60
    7.8958    43
    7.7750    26
    7.9250    23
    7.8542    21
    Name: Fare, dtype: int64



Second, the mean and percentiles:


{% highlight ruby %}
pd.concat([train[(train.Pclass==3)&(train.Embarked=='S')].Fare,test[(test.Pclass==3)& (test.Embarked=='S')].Fare]).describe()
{% endhighlight %}




    count    494.000000
    mean      14.435422
    std       13.118281
    min        0.000000
    25%        7.854200
    50%        8.050000
    75%       15.900000
    max       69.550000
    Name: Fare, dtype: float64



From this data, a sensible choice would be to assign 8.05. But again a little research, see [here](https://www.encyclopedia-titanica.org/titanic-victim/thomas-storey.html) shows that the passenger's ticket was bought at the same time as those of  Andrew Shannon [Lionel Leonard], August Johnson, William Henry Törnquist, Alfred Carver and William Cahoone Johnson. Let's find the fares for these passengers.


{% highlight ruby %}
train[train.Name.map(lambda x: 'Leonard' in x)]
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>Unk</td>
      <td>S</td>
    </tr>
    <tr>
      <th>121</th>
      <td>122</td>
      <td>0</td>
      <td>3</td>
      <td>Moore, Mr. Leonard Charles</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>A4. 54510</td>
      <td>8.0500</td>
      <td>Unk</td>
      <td>S</td>
    </tr>
    <tr>
      <th>179</th>
      <td>180</td>
      <td>0</td>
      <td>3</td>
      <td>Leonard, Mr. Lionel</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>Unk</td>
      <td>S</td>
    </tr>
    <tr>
      <th>248</th>
      <td>249</td>
      <td>1</td>
      <td>1</td>
      <td>Beckwith, Mr. Richard Leonard</td>
      <td>male</td>
      <td>37.0</td>
      <td>1</td>
      <td>1</td>
      <td>11751</td>
      <td>52.5542</td>
      <td>D35</td>
      <td>S</td>
    </tr>
    <tr>
      <th>386</th>
      <td>387</td>
      <td>0</td>
      <td>3</td>
      <td>Goodwin, Master. Sidney Leonard</td>
      <td>male</td>
      <td>1.0</td>
      <td>5</td>
      <td>2</td>
      <td>CA 2144</td>
      <td>46.9000</td>
      <td>Unk</td>
      <td>S</td>
    </tr>
    <tr>
      <th>626</th>
      <td>627</td>
      <td>0</td>
      <td>2</td>
      <td>Kirkland, Rev. Charles Leonard</td>
      <td>male</td>
      <td>57.0</td>
      <td>0</td>
      <td>0</td>
      <td>219533</td>
      <td>12.3500</td>
      <td>Unk</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>655</th>
      <td>656</td>
      <td>0</td>
      <td>2</td>
      <td>Hickman, Mr. Leonard Mark</td>
      <td>male</td>
      <td>24.0</td>
      <td>2</td>
      <td>0</td>
      <td>S.O.C. 14879</td>
      <td>73.5000</td>
      <td>Unk</td>
      <td>S</td>
    </tr>
    <tr>
      <th>758</th>
      <td>759</td>
      <td>0</td>
      <td>3</td>
      <td>Theobald, Mr. Thomas Leonard</td>
      <td>male</td>
      <td>34.0</td>
      <td>0</td>
      <td>0</td>
      <td>363294</td>
      <td>8.0500</td>
      <td>Unk</td>
      <td>S</td>
    </tr>
    <tr>
      <th>871</th>
      <td>872</td>
      <td>1</td>
      <td>1</td>
      <td>Beckwith, Mrs. Richard Leonard (Sallie Monypeny)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>1</td>
      <td>11751</td>
      <td>52.5542</td>
      <td>D35</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight ruby %}
 train[train.Name.map(lambda x: 'Johnson' in x)]
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>Unk</td>
      <td>S</td>
    </tr>
    <tr>
      <th>172</th>
      <td>173</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Miss. Eleanor Ileen</td>
      <td>female</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>Unk</td>
      <td>S</td>
    </tr>
    <tr>
      <th>302</th>
      <td>303</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. William Cahoone Jr</td>
      <td>male</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>Unk</td>
      <td>S</td>
    </tr>
    <tr>
      <th>597</th>
      <td>598</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. Alfred</td>
      <td>male</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>Unk</td>
      <td>S</td>
    </tr>
    <tr>
      <th>719</th>
      <td>720</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. Malkolm Joackim</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>347062</td>
      <td>7.7750</td>
      <td>Unk</td>
      <td>S</td>
    </tr>
    <tr>
      <th>869</th>
      <td>870</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Master. Harold Theodor</td>
      <td>male</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>Unk</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




{% highlight ruby %}
train[train.Name.map(lambda x: 'Tornquist' in x)]
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>271</th>
      <td>272</td>
      <td>1</td>
      <td>3</td>
      <td>Tornquist, Mr. William Henry</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0</td>
      <td>Unk</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



So, none of them paid for the ticket. The reason is because they were forced into the Titanic as another ship couldn't make the trip because of scheduling problems. So according to this the right value for fare should be 0.0. 

But note that there's an inconsistency. All of his shipmates have LINE under the ticket feature, but Storey, Mr. Thomas does not. I wonder why this happened? Lacking more information I think the value for Storey, Mr. Thomas should be the most common on, that is 8.05


{% highlight ruby %}
test.set_value(test.Fare.isnull(),'Fare',8.05);
{% endhighlight %}

### The Age Feature

We now turn to the missing Age values. A first option would be to replace them all by them for something similar to the case of Fare, but this is case dependent so, probably, not a reasonable thing to do; the amount of ages missing is more than 1/5. Instead, we will create a learning tool that predicts the age. We need to introduce some features first, that will help us later on as well.

## Some new Features

What other information can we extract from the features we already have? Let's look at them again. We can use the method [DataFrame.sample()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html) for this purpose.


{% highlight ruby %}
train.sample(n=4)
{% endhighlight %}




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>691</th>
      <td>692</td>
      <td>1</td>
      <td>3</td>
      <td>Karun, Miss. Manca</td>
      <td>female</td>
      <td>4.0</td>
      <td>0</td>
      <td>1</td>
      <td>349256</td>
      <td>13.4167</td>
      <td>Unk</td>
      <td>C</td>
    </tr>
    <tr>
      <th>195</th>
      <td>196</td>
      <td>1</td>
      <td>1</td>
      <td>Lurette, Miss. Elise</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17569</td>
      <td>146.5208</td>
      <td>B80</td>
      <td>C</td>
    </tr>
    <tr>
      <th>809</th>
      <td>810</td>
      <td>1</td>
      <td>1</td>
      <td>Chambers, Mrs. Norman Campbell (Bertha Griggs)</td>
      <td>female</td>
      <td>33.0</td>
      <td>1</td>
      <td>0</td>
      <td>113806</td>
      <td>53.1000</td>
      <td>E8</td>
      <td>S</td>
    </tr>
    <tr>
      <th>224</th>
      <td>225</td>
      <td>1</td>
      <td>1</td>
      <td>Hoyt, Mr. Frederick Maxfield</td>
      <td>male</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>19943</td>
      <td>90.0000</td>
      <td>C93</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Note that the Name Feature contains more information. The title: like Mr, Mrs, Dr, etc., and the surname that gives us info about groups of people traveling together (families).  The SibSp and Parch tell how large the family group is. The Cabin may be related to economical status. So let's create features that encode this.

### Features from text

There are many tools to mine info out of text, most of them use regular expresion. We load the python package to handle regular expressions. 


{% highlight ruby %}
import re
{% endhighlight %}

We want to break each of the names into parts and then categorize those. Note that the data before the comma is the last name and there's usually a tittle associated with the person mr, mrs, etc. We create a function to extract this.


{% highlight ruby %}
def extractInfo(word):
    pat=re.compile('(?P<surname>.+?), (?P<title>.*?)\.')
    A=re.search(pat,word)
    B=[A.group('surname'),A.group('title')]
    if A.group('title') in ['Mme']:
        B[1]='Mrs'
    elif  A.group('title') in ['Ms','Mlle']:
        B[1]='Miss'
    elif  A.group('title') in ['Don', 'Jonkheer','Master']:
        B[1]='Sir'
    elif A.group('title') in ['Dona', 'Lady', 'the Countess']:
        B[1]='Lady'
    elif A.group('title') in ['Capt', 'Col', 'Major', 'Dr', 'Officer', 'Rev']:
        B[1]='Officer'

    return B

{% endhighlight %}

The only nontrivial part here is how regular expressions work. Each set of parenthesis corresponds to a group, whose corresponding name is written after ?P in triangular brackets. ".+" means find any character except end of the line  and repeat at least once, ".*" means repeat zero or more times, and the ? after them means that is ungreedy (lazy). more detailed explanations [here](https://docs.python.org/3/howto/regex.html) and [here](http://www.regular-expressions.info/repeat.html). We input these new features next.


{% highlight ruby %}
#New features for train
temp=train.Name.map(extractInfo)
train['Surname']=temp.map(lambda x: x[0])
train['Title']= temp.map(lambda x: x[1])   

#New features for test
temp=test.Name.map(extractInfo)
test['Surname']=temp.map(lambda x: x[0])
test['Title']= temp.map(lambda x: x[1])   

{% endhighlight %}

Let's check what we got:


{% highlight ruby %}
train.Title.value_counts()+test.Title.value_counts()
{% endhighlight %}




    Mr         757
    Miss       264
    Mrs        198
    Sir         64
    Officer     23
    Lady         3
    Name: Title, dtype: int64



### Groups

The size of a family is relevant for the survival, we can think it this way families tend to survive or die together. So, we would like to keep track of this. We create two features to keep track of the size:

    -group_count: the number of people in the group.  
    -group_size:  'S' if group_count<2, 'M' is 2<= group_count<=4, 'L' if group_count>4. 


{% highlight ruby %}
train['group_count']=train.Parch + train.SibSp + 1
test['group_count']=test.Parch + test.SibSp + 1

train.set_value(train.group_count>0, 'group_size', 'M')
train.set_value(train.group_count<3, 'group_size', 'S')
train.set_value(train.group_count>4, 'group_size', 'L')

test.set_value(test.group_count>0, 'group_size', 'M')
test.set_value(test.group_count<3, 'group_size', 'S')
test.set_value(test.group_count>4, 'group_size', 'L');

{% endhighlight %}

### Normalizing the Fare Data 

Our next and final step before creating the classifier is to normalized fare, by first translating the mean and dividing by the standard deviation. This finishes our [preprocesing](http://machinelearningmastery.com/how-to-prepare-data-for-machine-learning/). We import the Standard Scaler from Sklearn. 


{% highlight ruby %}
from sklearn.preprocessing import StandardScaler 
scaler=StandardScaler()
{% endhighlight %}

[StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.fit_transform) has several methods:

 - fit(): computes the mean and standard deviation for later used.
 - transform(): transform the data using the mean and standard deviation already trained.
 - fit_transform(): does the previous two.

in versions <= 0.17 sklearn could take 1-dim'l arrays and use this to train and compute. Since then this has been deprecated. In our case, the data comes as a one-dimensional array, so we need to reshape it.


{% highlight ruby %}
fares = pd.concat([train.Fare,test.Fare]) # concatenate the train.Fare Data Series and the test.Fare Data Series

scaler.fit(fares.values.reshape(-1,1)) # Find the mean and std to be used later

#Normalizes the data for train and adds the feature NorFare.
train['NorFare'] = pd.Series(scaler.transform(train.Fare.values.reshape(-1,1)).reshape(-1), index=train.index) 

#Normalizes the data for test and adds the feature NorFare.
test['NorFare']=pd.Series(scaler.transform(test.Fare.values.reshape(-1,1)).reshape(-1), index=test.index) 
{% endhighlight %}

We now turn to the prediction of the Age Feature.

### Predicting Age

In order to predict the Age, we follow the following steps:

  - Combine the data from train and test.
  - Choose the relevant features to predict Age.
  - Separate the known Ages from the unknown ones.
  - Divide the known ones into the ones for training and the ones for testing.p
  - Create the classifier.
  - Test the classifier accuracy.
  - Predict the unknown ages.

#### Combining, choosing the relevant features, and separating

We use [pd.concat()](http://pandas.pydata.org/pandas-docs/stable/merging.html) to put the data together.


{% highlight ruby %}
allData= pd.concat([train,test])
{% endhighlight %}

We [drop](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html) the irrelevant features.


{% highlight ruby %}
allData.drop(labels=['PassengerId', 'Name', 'Cabin','Survived', 'Ticket', 'Fare','Surname'],axis=1,inplace=True)
{% endhighlight %}

Next we make the categorical variables into indicator variables. We use the method [get_dummies](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html) for this.


{% highlight ruby %}
allData=pd.get_dummies(allData, columns=['Embarked', 'Sex', 'Title', 'group_size'])
{% endhighlight %}

We obtain the known ages


{% highlight ruby %}
knownAges=allData[~allData.Age.isnull()]
{% endhighlight %}

We separate the features use for the model from the target feature.


{% highlight ruby %}
X=knownAges.drop(['Age'],axis=1)
Y=knownAges.Age
{% endhighlight %}

We split the data into training and testing.


{% highlight ruby %}
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=13)
{% endhighlight %}

We aim to use [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting). In this fork we use adaboost instead of xgboost. Keep in mind that [xgboost](http://xgboost.readthedocs.io/en/latest/) has better performance; go to the master branch for the use of xgboost. 


{% highlight ruby %}
from sklearn.ensemble import AdaBoostRegressor
model=AdaBoostRegressor(random_state=13)
{% endhighlight %}

In order to make sure we get thhe best classifier we use GridSearchCV to find the parameters that will give us the best fit.


{% highlight ruby %}
parameters={'n_estimators':range(2,20),'learning_rate':[x*0.1+0.1 for x in range(19)]}
ageClas=GridSearchCV(model,parameters)
ageClas.fit(X_train,Y_train );
{% endhighlight %}

We are ready to test how well we did.


{% highlight ruby %}
ageClas.score(X_test,Y_test)
{% endhighlight %}



---
 0.42300319376063289
 
---


This is not pretty good, but it would be difficult to acchieve better accuracy with such a small data set and not much relevant data to age. Let's see what features were more important. We first find what was the best classifier that the GridSSearch found.


{% highlight ruby %}
ageClas.best_params_
{% endhighlight %}




    {'learning_rate': 0.1, 'n_estimators': 4}



Then, we create the classifier associated with those parameters and see what were the important features. 


{% highlight ruby %}
model2=AdaBoostRegressor(learning_rate=0.1,n_estimators=4,random_state=13)
model2.fit(X_train,Y_train)
model2.feature_importances_
{% endhighlight %}




    array([  7.03742379e-03,   9.16024507e-02,   2.34270827e-01,
             0.00000000e+00,   1.07496543e-02,   7.87789897e-06,
             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
             0.00000000e+00,   0.00000000e+00,   2.97504208e-01,
             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
             3.58827559e-01,   0.00000000e+00,   0.00000000e+00,
             0.00000000e+00])



this gives the third and twelve features as the most important ones. This correspond to Siblings, the fact if the person have a title of Mr. and if it has a title of Sir. Let's use this regressor to give the unknown values to Age.


{% highlight ruby %}
pred = ageClas.predict(allData[allData.Age.isnull()].drop('Age', axis=1))
allData.set_value(allData.Age.isnull(), 'Age', pred);
{% endhighlight %}

We can now compare the density plot for Age with the one we had before.


{% highlight ruby %}
fig = plt.figure(figsize=(18, 4)); # We create our canvas.
alpha = 0.7; # How transparent curves will be.

# Subdivides the canvas in one row and two columns and says we are plotting at the one with coordinates (0,0)
ax0 = plt.subplot2grid((1,1), (0,0)) 


# Obtains the feature Age from the data set, then it plots this as a density function using the kde type for DataFrame.plot()
train.Age.plot(kind='kde', color='#FA2379', label='train', alpha=alpha)
test.Age.plot(kind='kde', label='test', alpha=alpha)
allData.Age.plot(kind='kde', label='All Data after prediction of Ages', alpha=alpha)

plt.legend(loc='best');
{% endhighlight %}

<img src="{{ '/img/Titanic_files/Titanic_136_0.png' | prepend: site.baseurl }}" alt=""> 


Finally, let's put this results back in the train and test dataFrames.


{% highlight ruby %}
train.Age=allData.Age[:891]
test.Age=allData.Age[891:]
{% endhighlight %}

We can now move into creating our classifiers for predicting survival.

## Predicting Survival

Before predicting survival we need to make the data ready to train the classifiers.

### Normalizing

We have already normalized fare. But we still need to normalize Age and group_count. 


{% highlight ruby %}
allData['NorAge'] = pd.Series(scaler.fit_transform(allData.Age.values.reshape(-1,1)+0.0).reshape(-1), index=allData.index)
allData['NorGroup_count'] = pd.Series(scaler.fit_transform(allData.group_count.values.reshape(-1,1)+0.0).reshape(-1), index=allData.index)

train['NorAge']=allData.NorAge[:891]
train['NorGroup_count']=allData.NorGroup_count[:891]

test['NorAge']=allData.NorAge[891:]
test['NorGroup_count']=allData.NorGroup_count[891:]
{% endhighlight %}

### Gender as values

We encode female gender as 1 and male as 0.


{% highlight ruby %}
train.Sex=np.where(train.Sex=='Female',1,0)
test.Sex=np.where(test.Sex=='Female',1,0)
{% endhighlight %}

We get rid of the features we need no more and the Surname feature since we won't use it.


{% highlight ruby %}
train.drop(labels=['PassengerId', 'Surname','Name', 'Cabin', 'Ticket', 'Age', 'Fare'],axis=1,inplace=True)
test.drop(labels=['Name', 'Cabin', 'Surname','Ticket', 'Age', 'Fare'],axis=1,inplace=True)
{% endhighlight %}

### Indicator/Dummy Features

As we did above with Age, we need to make the features numeric, we can do this with Dummy Features. (Note that as we are keeping the surname feature the dimensionality should increase considerably)


{% highlight ruby %}
train=pd.get_dummies(train, columns=['Embarked', 'Title', 'group_size'])
test=pd.get_dummies(test, columns=['Embarked', 'Title', 'group_size'])
{% endhighlight %}

### Some tools for graphing

In order to see how the classifiers behave we will use some extra tools. First the [learning_curve](http://scikit-learn.org/stable/modules/generated/sklearn.learning_curve.learning_curve.html#sklearn.learning_curve) method. It Determines cross-validated training and test scores for different training set sizes.


{% highlight ruby %}
from sklearn.model_selection import learning_curve
{% endhighlight %}

Then we follow the [sklearn docs](http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html) to create a function that graphs the learning cure for a given classifier. 


{% highlight ruby %}
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.2, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

{% endhighlight %}

### Choosing the right parameters

The choice of a classifier usually depends on a choice of parameters. There are automatic ways of doing this, we use before above when coming up with a classifier for Age, that is GridSearchCV. The next function takes a classifier, a range of values for the parameters, some training data, a scoring method and returns the classifier associated with the best parameters.


{% highlight ruby %}
def bestClass(clas,parameters,X_train, Y_train, scoring):
    grid = GridSearchCV(clas, param_grid=parameters, scoring=scoring)
    grid.fit(X_train,Y_train)
    return grid.best_estimator_
{% endhighlight %}

Note the scoring function, we talk about this next.

### Measuring accuracy.

Next, we import a method to measure accuracy, we import [accuracy_score](http://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score) which finds the percentage of correct predictions.


{% highlight ruby %}
from sklearn.metrics import accuracy_score, make_scorer
{% endhighlight %}

Even though this is a function that computes accuracy, we need to wrapp it to make it a scoring function. That is what [make_scorer](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer) does. 


{% highlight ruby %}
scoring = make_scorer(accuracy_score, greater_is_better=True)
{% endhighlight %}

### Splitting Data

We split the data into training and test as we did with Age above.


{% highlight ruby %}
X = train.drop(['Survived'], axis=1)
Y = train.Survived
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=13)
{% endhighlight %}

We will create different classifiers and see how they perform. 

## K-nearest Neighbors (KNN)

The [K-nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) is a classifier that basically outputs the average of the values of its k nearest neighbors.  You can find more info about it [here].


{% highlight ruby %}
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(weights='uniform')
parameters = {'n_neighbors':[3,4,5], 'p':[1,2]}
clf_knn = bestClass(KNN, parameters, X_train, Y_train, scoring)
{% endhighlight %}

We can see how well we did.


{% highlight ruby %}
accuracy_score(Y_test, clf_knn.predict(X_test))
{% endhighlight %}




    0.80717488789237668



And look at a graph on how the accuracy improves against the training data size.


{% highlight ruby %}
plot_learning_curve(clf_knn, 'KNN', X, Y, cv=4);
{% endhighlight %}


<img src="{{ '/img/Titanic_files/Titanic_177_0.png' | prepend: site.baseurl }}" alt=""> 


## Random Forest

The [Random Forest](https://en.wikipedia.org/wiki/Random_forest) classifier is an ensemble kind of classifier, made out of decision trees. It usually helps with the overclassifying problem that decision trees tend to have. More info [here]. 


{% highlight ruby %}
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=13, criterion='entropy', min_samples_split=8, oob_score=True)
parameters = {'n_estimators':[300,500], 'min_samples_leaf':[10,12]}
clf_rfc1 = bestClass(rfc, parameters, X_train, Y_train, scoring)
{% endhighlight %}

We can see how well Random Forest does.


{% highlight ruby %}
accuracy_score(Y_test, clf_rfc1.predict(X_test))
{% endhighlight %}




    0.8340807174887892



The learning curve graph is 


{% highlight ruby %}
plot_learning_curve(clf_rfc1, 'Random Forest', X, Y, cv=4);
{% endhighlight %}

<img src="{{ '/img/Titanic_files/Titanic_184_0.png' | prepend: site.baseurl }}" alt=""> 

This isn't pretty good, can we improve this? Let's check how many features contributed more than 0.1%.


{% highlight ruby %}
clf_rfc1.feature_importances_[clf_rfc1.feature_importances_>0.001]
{% endhighlight %}




    array([ 0.09015719,  0.02417626,  0.00830836,  0.03181073,  0.13527542,
            0.08537893,  0.03187369,  0.00919137,  0.00857778,  0.01750316,
            0.11424149,  0.28841687,  0.10881213,  0.00761623,  0.02074611,
            0.01201022,  0.00550017])



We now create a ramdon forest only using those features. Let's first select the corresponding columns.


{% highlight ruby %}
cols = X_train.columns[clf_rfc1.feature_importances_>=0.001]
{% endhighlight %}

We create the classifier now


{% highlight ruby %}
rfc = RandomForestClassifier(random_state=13, criterion='entropy', min_samples_split=5, oob_score=True)
parameters = {'n_estimators':[300,500], 'min_samples_leaf':[10,12]}
clf_rfc2 = bestClass(rfc, parameters, X_train[cols], Y_train, scoring)
{% endhighlight %}

Our new accuracy is


{% highlight ruby %}
accuracy_score(Y_test, clf_rfc2.predict(X_test[cols]))
{% endhighlight %}




    0.82511210762331844



Wich is much better, let's look at the learning curve for this model.


{% highlight ruby %}
plot_learning_curve(clf_rfc2, 'Random Forest', X[cols], Y, cv=4);
{% endhighlight %}


<img src="{{ '/img/Titanic_files/Titanic_194_0.png' | prepend: site.baseurl }}" alt=""> 



## Logistic Regression

The [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) classifier predicts the probability of the outcome. More detailed info [here]. 


{% highlight ruby %}
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(random_state=42, penalty='l1')
parameters = {'C':[0.3,0.5,0.8]}
clf_lg = bestClass(lg, parameters, X_train, Y_train, scoring)
{% endhighlight %}

We get an accuracy of


{% highlight ruby %}
accuracy_score(Y_test, clf_lg.predict(X_test))
{% endhighlight %}




    0.83856502242152464



And a learning curve:


{% highlight ruby %}
plot_learning_curve(clf_lg, 'Logistic Regression', X, Y, cv=4);
{% endhighlight %}


<img src="{{ '/img/Titanic_files/Titanic_201_0.png' | prepend: site.baseurl }}" alt=""> 



## Support Vector Machine (SVM)

The [SVM](https://en.wikipedia.org/wiki/Support_vector_machine) classifier finds the best hyperplane separating the points on the class survived=True from the class survived=False. As this is not always possible the classifier usually relies on the Kernel trick, the problem being that it increases dimentionality. This classifier should run quite slowly and usually some data preparation needs to be used to reduce dimensionality (PCA for example) more info [here].


{% highlight ruby %}
from sklearn.svm import SVC
svc = SVC(random_state=42, kernel='poly', probability=True)
parameters = {'C': [30,35], 'gamma': [0.0055,0.001], 'coef0': [0.1,0.2],
              'degree':[2,3]}
clf_svc = bestClass(svc, parameters, X_train, Y_train, scoring)
{% endhighlight %}

Let's check the accuracy


{% highlight ruby %}
accuracy_score(Y_test, clf_svc.predict(X_test))
{% endhighlight %}




    0.8340807174887892



And the learning curve


{% highlight ruby %}
plot_learning_curve(clf_svc, 'SVC', X, Y, cv=4);
{% endhighlight %}


<img src="{{ '/img/Titanic_files/Titanic_208_0.png' | prepend: site.baseurl }}" alt=""> 



## Voting Classifier

The [voting classifier](http://scikit-learn.org/stable/modules/ensemble.html#voting-classifier) submits the classification to vote among other classifiers, more info [here]. It is part of the ensemble classifiers, we use the classifiers we have already and submit them to this.


{% highlight ruby %}
from sklearn.ensemble import VotingClassifier
clf_vc = VotingClassifier(estimators=[ ('lg', clf_lg), ('svc', clf_svc), 
                                      ('rfc1', clf_rfc1),('rfc2', clf_rfc2), ('knn', clf_knn)], 
                          voting='hard', weights=[1,1,1,1,2])
clf_vc = clf_vc.fit(X_train, Y_train)
{% endhighlight %}

We get an accuracy of 


{% highlight ruby %}
accuracy_score(Y_test, clf_vc.predict(X_test))
{% endhighlight %}




    0.82959641255605376



And a learning curve 


{% highlight ruby %}
plot_learning_curve(clf_vc, 'Voting Classifier', X, Y, cv=4);
{% endhighlight %}

<img src="{{ '/img/Titanic_files/Titanic_215_0.png' | prepend: site.baseurl }}" alt=""> 


# Conclusion

We can create classifiers that predict surival with an accuracy > .8. 

# Submission


{% highlight ruby %}
test.head(n=2)
{% endhighlight %}


{% highlight ruby %}
PassengerId = test.PassengerId
test.drop('PassengerId', axis=1, inplace=True)
{% endhighlight %}


{% highlight ruby %}
def submission(model, fname, X):
    ans = pd.DataFrame(columns=['PassengerId', 'Survived'])
    ans.PassengerId = PassengerId
    ans.Survived = pd.Series(model.predict(X), index=ans.index)
    ans.to_csv(fname, index=False)
submission(clf_vc,'submission.csv',test)

{% endhighlight %}


{% highlight ruby %}

{% endhighlight %}
