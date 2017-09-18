---
layout: post
title:  "Study the Data"
date:   2017-09-13
category: general
---

In this and the following posts we study the Amazon reviews dataset and proceed to build models to predict the Sales Rank feature. These post are the result of the HackOn(Data) competition of 2017. 

**Team**: Hari, Con, and Felipe.

# [HackOn(Data)](https://hackondata.com/2017/index.html#home)

## Competition Challenge

Welcome to our project. We have decided to do a simple Forecasting on the salesRank, more precesilly, 

**The problem:** Predict salesRank given the reviews information. 

**Hypothesis:** The reviews are an indicative of how well a product is selling and contain enough information for a forecast.

**NOTE:** This notebook is intended to show the overall ideas and, in order to make it easier to read, we have sacrificed speed. 

**NOTE 2:** For the code and details refer to the [github repo](https://github.com/Scy1505/hackon_data).

This post are organized as follows:

- **Part 1**: Studying the data.
- **Part 2**: Cleaning and Preparing the Data. 
- **Part 3**: Feature Engineering.
- **Part 4**: Model Building.
- **Part 5**: Evaluating the results.
- **Part 6**: Scalability.


# Part 1- Studying the Data

The amazon review data set has been broadly studied in order to mine for data. We start by giving some general ideas on how the data looks like. 

### **The Amazon Review Data**

This dataset contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014.

This dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs).

### **Q&A Data**

This dataset contains Questions and Answers data from Amazon, totaling around 1.4 million answered questions.

This dataset can be combined with Amazon product review data, by matching ASINs in the Q/A dataset with ASINs in the review data. The review data also includes product metadata (product titles etc.).

### **Credits:**

-  **R. He, J. McAuley**. Modeling the visual evolution of fashion trends with one-class collaborative filtering. WWW, 2016 J. 
- **McAuley, C. Targett, J. Shi, A. van den Hengel**. A Image-based recommendations on styles and substitutes. SIGIR, 2015

The scripts provided at [Julian McAuley's website](http://jmcauley.ucsd.edu/data/amazon/) allow us to load the different data sets we are going to study. More concretely, we load the set of all Video Games reviews, the set of all Video Games reviews for the games that have at least 10 reviews. The meta data associated to the products as well as the Q/A of these products. For the details, check at their website. 

The whole set of reviews consists of 1324753 reviews, unfortunately, only 52158 of them correspond to products that have at least 10 reviews, that's just about 4% of the total dataset. 

The features in the review dataset are of sometimes redundant, and we usually just ignore the superflous features. For example, the unixReviewTime (which is just the number of seconds since 1970) and the reviewDate both have the same info, for simplicity we take the unixReviewTime and transform it to the number of days since the day of the first review (May 1996).


To get more info on the reviews, we compute both the character lenght, and the number of words in the reviews. 

{% highlight ruby %}
df_reviews['review_length']=df_reviews.reviewText.apply(len)
df_reviews['nb_words']=df_reviews.reviewText.apply(lambda x: len(x.split()))
{% endhighlight %}

Let's look at the histograms of these features on the datasets. First for the whole dataset.


<center>
<img src="{{ '/assets/img/Study_the_data_files/Study_the_data_14_0.png' | prepend: site.baseurl }}" alt=""> 
</center>

And then only for the dataset with at least 10 reviews. 

<center>
<img src="{{ '/assets/img/Study_the_data_files/Study_the_data_15_0.png' | prepend: site.baseurl }}" alt=""> 
</center>

Note the diffrence in scales in the y-axis. Again, this is due to the diffrence on size of the datasets.

The first insights that we get are that most people write shorter reviews. 

Let's create similar graphs for the number of words in the reviews. The whole dataset. 


<center>
<img src="{{ '/assets/img/Study_the_data_files/Study_the_data_16_0.png' | prepend: site.baseurl }}" alt=""> 
</center>

and for the ones with at least 10 reviews. 

<center>
<img src="{{ '/assets/img/Study_the_data_files/Study_the_data_17_0.png' | prepend: site.baseurl }}" alt=""> 
</center>

The information is similar to the ones in character lenght and we don't gain a lot more insight. 

As our next step we would like to understand how the reviews are distributed in time. The first that we note is that only 32 reviews were produced in the first two years, so we ignore these and assume the first review happened in 1998 instead of 1996. Let's create the histograms for this variable and the datasets.



<center>
<img src="{{ '/assets/img/Study_the_data_files/Study_the_data_24_0.png' | prepend: site.baseurl }}" alt=""> 
</center>



<center>
<img src="{{ '/assets/img/Study_the_data_files/Study_the_data_25_0.png' | prepend: site.baseurl }}" alt=""> 
</center>




This graph shows something interesting, the number of reviews increase every year about the same time. That is we have a seasonal behavior. The reason is video games summer sales, driven probably by events like E3. From these graphs we can gather an important piece of data, there is a significant corelation between number of reviews and the date of the review.


Next, let's see about the distributtion of the ratings.


<center>
<img src="{{ '/assets/img/Study_the_data_files/Study_the_data_27_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


<center>
<img src="{{ '/assets/img/Study_the_data_files/Study_the_data_28_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


The phenomena here is quite interesting, it says that, at least for Video Games, people reviews more often the products they like. Furthermore, comparing the two graphs we see that when a product doesn't have many reviews is slighly more likely to have a poor rating.

 The next part of our analysis focuses on the salesRank. we first obtain the Video Games salesRank info, which is encoded on the salesRank feature of the metadata.


{% highlight ruby %}
RELEVANT_KEY='Video Games'
def helper(dictionary):
    try:
        return dictionary[RELEVANT_KEY]
    except:
        return float('NaN')
df_meta['salesRank']=df_meta.salesRank.map(lambda x: helper(x))
df_meta=df_meta[ df_meta.salesRank.notnull()]
{% endhighlight %}

We merge the dataset with group by products and agregate by product to find some general info.


{% highlight ruby %}
df_products_reviews=df_meta.merge(df_reviews, on='asin')
df_products=df_products_reviews[['asin','reviewText','salesRank','overall']].groupby('asin').agg({
    'reviewText':'count',
    'salesRank':'min',
    'overall':'mean'})
df_products.columns=['overall','numberOfReviews','salesRank']
{% endhighlight %}

let's now look at the histogram on the number of reviews for each product.

<center>
<img src="{{ '/assets/img/Study_the_data_files/Study_the_data_36_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


A large part of the data has just a few reviews. Let's take a closer look to the data between 10 and 60 reviews.


<center>
<img src="{{ '/assets/img/Study_the_data_files/Study_the_data_38_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


This looks more reasonable for study, as we will see later, there is more info here than in the rest. We next study the salesRank feature as such.


<center>
<img src="{{ '/assets/img/Study_the_data_files/Study_the_data_39_1.png' | prepend: site.baseurl }}" alt=""> 
</center>

<center>
<img src="{{ '/assets/img/Study_the_data_files/Study_the_data_39_2.png' | prepend: site.baseurl }}" alt=""> 
</center>

This reveals to us several interesting phenomena about the dataset, first it is not balanced, not only there are many ranks without an associated product, but there are several products that have the same rank. This imbalances are the main difficulties to overcome in the predictions of the salesRank.


Finally, let's compare the number of reviews with the salesRank. We can see some relations begin to show up, like the fact that a large ammount of reviews imply better ranking, but it is clear that this is not enough for prediction. 

<center>
<img src="{{ '/assets/img/Study_the_data_files/Study_the_data_40_0.png' | prepend: site.baseurl }}" alt=""> 
</center>

Let's see what the combined information of the number of reviews and the average rating gives. T


<center>
<img src="{{ '/assets/img/Study_the_data_files/Study_the_data_44_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


Which in particular shows that low rating (~1) ends up in poor salesRank. Which again is not enough, in the next post we will produce more features and use fancier models to make predictions.