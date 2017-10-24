---
layout: post
title:  "Preparing the data"
date:   2017-09-17
category: general
---

In our [previous post]({{ site.baseurl }}{% post_url 2017-9-13-Study_the_data %}) we did a quick study of the data set provided to us, in this post we clean the data and do some feature engineering which will prepare our dataset for training. This process is a long and important one. 

**Note**: For the code and details refer to the [github repo](https://github.com/Scy1505/hackon_data).


# Part 2- Cleaning and preparing the Data

After loading the data we study the different features in each of the datasets.


## Features from the meta file

### SalesRank

Note that the dataframe that contains the salesRank is df_meta. In this data there are two possible cases where the values described are not helpful. The first one is a NaN value, and we can use is null method to deal with this, the second one is the case of a dictionary no containing the relevant key. We use a helper function to help us with this.


{% highlight ruby %}
RELEVANT_KEY='video games'
def helper(dictionary):
    try:
        return dictionary[RELEVANT_KEY]
    except:
        return float('NaN')
{% endhighlight %}


{% highlight ruby %}
df_meta['salesRank']=df_meta.salesRank.map(lambda x: helper(x))
{% endhighlight %}

It is now straighforward to find out that 5675 products don't have the desired SalesRank feature, this corresponds to about 11% of all the products. As this is useless for our purposes we remove these rows 


### imUrl

This features contains the web address to an image. As ee won't be using the images, so we drop this column.

### Title and brand

A couple lines of code give us that only 0.22% of the products have title, and only 0.11% of the products have brand. That is, these features are totally useless, so we drop them as well.


### Price

The feature price is definitely relevant for the  forecasting, and  88.25% of the products have it. We have a couple of choices on how to deal with the missing values. 

- Fill it in with a natural replacement (average, zero, etc.).
- Drop the products.
- Create a categorical feature to separate the cases of having and no having the price.

We opt for the second option. After this cleaning we find that there are 39959 products left.


### Categories

The categories feature comes as a list of lists, let's first find out how many different categories are there.


{% highlight ruby %}
categories=set()
for list_cats in df_meta.categories:
    for list_cat in list_cats:
        categories=categories.union(set(list_cat))
categories=list(categories)
print("There are %d categories"%len(categories))
{% endhighlight %}

    There are 334 categories


we use these as a categorial variable, first we create a unique list of categories for each product


{% highlight ruby %}
def helper2(list_cats):
    cats=set()
    for list_cat in list_cats:
        cats=cats.union(set(list_cat))
    return list(cats)
{% endhighlight %}


{% highlight ruby %}
df_meta['categories']=df_meta.categories.apply(helper2)
{% endhighlight %}

and then we use the [MultiLabelBinarizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) from sklearn to do the encodding. 


{% highlight ruby %}
mlb=MultiLabelBinarizer()
mlb.fit(df_meta.categories)
categories_df=pd.DataFrame(mlb.transform(df_meta.categories),columns=list(mlb.classes_))
df_meta.reset_index(drop=True,inplace=True)
df_meta=df_meta.merge(categories_df,left_index =True,right_index =True)
{% endhighlight %}

we drop the categories column since we need it no more.

### Related and decription


We won't be using this either, so we remove them. 

**Note:** Potentially, there is relevant information here. Since there may be a correlation in the salesRank for related products. 

## Features from the reviews file

### reviewText

This feature, whose values contain the review will be used for sentiment analysis. We come back to this when building features. We note that there are 304 empty reviews on the dateset with all reviews and 3 empty ones for products that have at least 10 reviews. We drop those reviews.

### summary

We won't be using this feature, so we drop it as well.

### reviewTime

This feature is redundant since we have unixReviewTime, so we also get rid of it.

### ReviewerName
This is also (a little) redundant, since we have the reviewerID, we drop it. 
**Note:** In theory we could do more with the names than we the ID. The reason is that we could use the names to try to guess how trustworthy/real a reviewr is. 

### Helpful
The helpful feature comes as a pair of integers, the first one represents the number of people who found the review helpful, the second one is the number of people giving an opinion on the review. We separate this info into two features.


{% highlight ruby %}
df_reviews[['wasHelpful','helpfulFeedback']] = pd.DataFrame(df_reviews.helpful.values.tolist())
{% endhighlight %}

and drop the helpful feature which is now redundant.


## Features from QA data
We clean a couple features from the question and answer dataset. 

### answerType

There are three possibilities for the kind of answer Y,N, and ? so we add this feature.


{% highlight ruby %}
df_qa[['Y_answer','N_answer','?_answer']]=pd.get_dummies(df_qa.answerType)
{% endhighlight %}


And get rid of the superfluous feature answerType.

# Part 2 - Feature engineering

In this part we build our features, this will be done in the following order.


- Use sentiment analysis to give a sentiment score to the reviews. 

- helpfulnes features: increases the confidence on the review.

- ReviewerID features: quality of reviewers correlated to quality of reviews.

- ReviewTime features: To create a time series for the (cummulative) number of reviews.


### Sentiment Analysis on reviews

The idea of finding a sentiment score out of the reviews is to normalized the score given by the reviewers. That is different people may give different scores and write almost the same review. The sentiment analysis will then give the same score to similar reviews.

We use [fasttext](https://github.com/facebookresearch/fastText) from facebook to create an score. This was build using 

A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification

For this we need to save the data we need to a text file, since that is the signature of the training method of fasttext.


{% highlight ruby %}
df_sentiment=df_reviews.loc[:,('reviewText','overall')]
df_sentiment['overall']=df_sentiment.overall.apply(lambda x: '__label__'+str(int(x)))
df_sentiment.to_csv(r'./data_/data_for_sentiment.txt', header=None, index=None, sep=' ', mode='a')
{% endhighlight %}

Now, that the data is prepared for procesing, we can use the classifier.


{% highlight ruby %}
classifier = fasttext.supervised('./data_/data_for_sentiment.txt', 'model', label_prefix='__label__')
{% endhighlight %}

and add this feature to the reviews dataframes


{% highlight ruby %}
df_reviews['sentimentScore']=df_reviews.reviewText.apply(
    lambda x:int(classifier.predict([x])[0][0][-1]))
{% endhighlight %}

as that was our use of the textReview feature we can get rid of it. Now, let's compare the distribution of the scores given by the overall score and the sentimentScore.



<center>
<img src="{{ '/assets/img/Preparing_the_data_files/Preparing_the_data_80_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


We can see see that the scores move towards the extremes after sentiment. 

We also pass a jugement to the review given the its lenght by the formula.

$$ \text{review_judgment}=\text{overall}*\frac{\text{review_length}-3}{2}. $$


{% highlight ruby %}
df_reviews['review_length']=df_reviews.reviewText.apply(lambda x:len(x))
df_reviews['review_judgment']=df_reviews.overall*(df_reviews.review_length-3)/2

{% endhighlight %}


We finish by droping the reviewText feature.

### ReviewerID

We want to measure the quality of a reviews. One clear option is already given to us via the helpful features. As we want more, we add new features that come from the reviewerID. In order to obtain this features we create a new dataframe, one obtained by aggregating information on the reviewers. 


{% highlight ruby %}
df_reviewers=df_reviews.merge(df_meta[['asin','price']],on='asin').groupby('reviewerID').agg({
                              'overall':['min','max','mean', 'std','count'], # details of ratings for this reviewer
                              'unixReviewTime':['min','max'], # date of first and last review
                              'wasHelpful': ['sum', 'min', 'max', 'mean', 'std'],
                              'helpfulFeedback':['sum', 'min', 'max', 'mean', 'std'],
                              'price': ['sum', 'min', 'max', 'mean', 'std'], # average price of items reviewed etc
                             })
{% endhighlight %}


We add this features to df_reviews.


{% highlight ruby %}
#As we are choosing inner, this procedure automatically removes the items for which there's not price
df_reviews=df_reviews.merge(df_reviewers, left_on='reviewerID',right_index=True)
{% endhighlight %}

as we won't be using the reviewerID anymore, we get rid of this feature.


## Data aggregation

Before we continue with the feature generation, we need to agregate the data by product. This will make the features to be a time series, where the time is controled by the UnixReviewTime. After some low level changes we can create a dictionary of aggregating operation for the different features we care about.


{% highlight ruby %}
agg_ops_dict={feature_name:['sum', 'min', 'max', 'mean', 'std'] for feature_name in df_reviews.columns if feature_name!='asin'}
agg_ops_dict['unixReviewTime']+=[lambda x:(list(x))]
agg_ops_dict['overall']+=[lambda x:(list(x))]
agg_ops_dict['sentimentScore']+=[lambda x:(list(x))]
agg_ops_dict['wasHelpful']+=[lambda x:(list(x))]
agg_ops_dict['helpfulFeedback']+=[lambda x:(list(x))]
{% endhighlight %}

and we aggregate using this dictionary. 

{% highlight ruby %}
df_products=df_reviews.groupby('asin').agg(agg_ops_dict)
{% endhighlight %}


### helpful

In order to incorporate the helpful score into the product, we create two variables defined below:

$$ \text{helpfulOverall} = \sum \text{overall}_i \cdot \frac{\text{wasHelpful}}{\text{helpfulFeedback}} $$ 

and 

$$ \text{helpfulSentiment} = \sum \text{sentiment_score}_i \cdot \frac{\text{wasHelpful}}{\text{helpfulFeedback}} $$ 


where the sums run over the $$i$$ such that helpful$$_i\neq 0.$$ After incorporting these features we drop the list obtained from the data aggregation.


**REMARK:** The reader should note that there is a faster way to compute the last two features, readly starting from the df_reviews dataframe building the individual multiplications and then aggregating. We opted for the (significantly) slower way to make the notebook easier to read.

### The hotness and density features.

We introduce a family of features designed to encode the behavior fo the cumulative function of number of reviews. We need to do some cleaning first.

As the unixReviewTime is the number of seconds since 1970, we simplify this so it keeps track of the number of days instead. We also create a feature to encode the first day that the product got a review, we also create a list with the days after the first review that each review happened


we add a feature called productLife, keeping track of the number of days during which the product got reviews. We also added the feature for the number of reviews, nunmberReviews.


{% highlight ruby %}
df_products['productLife']=df_products.daysSinceFirstReview.apply(lambda x:x[-1])
df_products['numberReviews']=df_products.daysSinceFirstReview.apply(lambda x: len(x))
{% endhighlight %}

The next set of features is based on the following function.

{% highlight ruby %}
def create_cumulative(days):
    X=[days[0]]
    Y=[0]
    current=0
    for day in days:
        if day==X[-1]:
            Y[current]+=1
        else:
            X.append(day)
            Y.append(Y[-1]+1)
            current+=1
    return X,Y
{% endhighlight %}


To illustrate the next feature we are going to create, we graph the cummulative function for a sample set of reviews.


<center>
<img src="{{ '/assets/img/Preparing_the_data_files/Preparing_the_data_124_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


Let's make the following assumption:

**Assumption: The "better" a product is, the more reviews it would have when measure over periods of time**

In order to make this concrete, we can create new invariant. 

**hotness** = area under the staircase formed by the curve in the interval where it is defined.

Similarly we encode information of the product by its density, which relies on the assumption

**Assumption: The change on reviews per day is an indicative of the performance of the product.**

**density** = slope formed between the first point and a future point.

We use the following function to compute some statistics on the hotness and density features. 

{% highlight ruby %}
def compute_hotness(days):
    X,Y=create_cumulative(days)
    area_list=[0]
    area=0
    for i in range(1,len(X)):
        area+=(X[i]-X[i-1])*Y[i-1]
        area_list.append(area)
    area+=Y[-1]
    area_list.append(area)
    
    area_list = np.array(area_list)
    return area_list.max(),area_list.mean(),area_list.std()
{% endhighlight %}

and add these stats to the dataframe. 

## Counting questions

We add features on how many question of each type are there associated with each product. 

{% highlight ruby %}
df_qa_agg = df_qa.groupby('asin').agg({
                              'asin':['count'], 
                              'unixTime':['min','max'],
                              'Y_answer': ['sum'],
                              'N_answer': ['sum'],
                              '?_answer': ['sum']
                             })
{% endhighlight %}


This concludes our feature creation, our next step is to set up the data for training.


# Creating the X and y's

We want to create a dataset containing the features our models will use. We start by merging our meta_data to the products.


{% highlight ruby %}
Xy_df=df_products.merge(df_meta, left_index=True,right_on='asin')
Xy_df=Xy_df.merge(df_qa_agg,how='outer',left_on='asin',right_index=True).fillna(0)
{% endhighlight %}


Our goal is to be able to predict the rank, as this is a difficult feature, we do the best thing, that is we predict to which rank range the product belongs to, so we need to create bins each containing the same amount of data points. We need some auxiliary functions.


{% highlight ruby %}
salesRanks=np.sort(Xy_df.salesRank.as_matrix())
len_salesRank,=np.sort(Xy_df.salesRank.as_matrix()).shape
def find_breakpoints(nb_bins):
    step=len_salesRank//nb_bins
    return [salesRanks[i] for i in range(step-1,len_salesRank,step)]
{% endhighlight %}

We want to add many possible outcomes for the number of bins, so we create different target features. We need an auxiliary function for this.


{% highlight ruby %}
def binning(breakpoints,a):
    i=0
    while i< len(breakpoints):
        if a>breakpoints[i]:
            i+=1
        else:
            return str(i+1)+"_out_of_"+str(len(breakpoints))
    return str(i+1)+"_out_of_"+str(len(breakpoints))
{% endhighlight %}

We now add the different ammount of buckets, we go from 2 to 10.


{% highlight ruby %}
for i in range(2,11):
    breakpoints=find_breakpoints(i)
    Xy_df['salesRank_Category_of_'+str(i)]=Xy_df.salesRank.apply(lambda x: binning(breakpoints,x))
{% endhighlight %}

as all our desired data is here, we save this dataframe.


{% highlight ruby %}
pickle.dump(Xy_df, open('./Xy_df_videoGames', 'wb'))
{% endhighlight %}

To conclude we notice that there are about 40 thousand products, and we created about 500 features. 
