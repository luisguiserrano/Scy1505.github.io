---
layout: post
title:  "The models"
date:   2017-09-20
category: general
---

This is the last part on the series of the Amazon review dataset. In our [previous post]({{ site.baseurl }}{% post_url 2017-9-17-Preparing_the_data %}) we ready the data for training. In this post we build, train, and evaluate our models. 


# Part 4 - Model Building and the results

After loading the  data we created in our previous post. We note that there are about 500 features and 8 possible outcomes. There are a couple features that are not numerical, like daysSinceFirstReview, so we drop them. It is easy then to separate the features from the predictions.


{% highlight ruby %}
y_cats={}
for i in range(2,11):
    y_cats[i]=Xy_df['salesRank_Category_of_'+str(i)].as_matrix()
    
X_df=Xy_df.drop(['salesRank','asin']+['salesRank_Category_of_'+str(i) for i in range(2,11)],axis=1)
X=X_df.as_matrix()

{% endhighlight %}


# Some models 

We are going to use two different assembler classifiers to create the models.

### Random Forest

The first model we try is Random Forest, we tune n_estimators and max_features hyperparameters using a CVGridSearch approach. This will be running for a while. We also save our best models on file for future use. Keep in mind that these models are quite large and it may take a couple GB of space. 

{% highlight ruby %}
parameters = {'n_estimators':[10,100,500,1000], 'max_features':[10, 20,30,40]}
{% endhighlight %}


{% highlight ruby %}
scores_RandomForest=[]

for i in range(2,11):
    
    print('Training for %d bins'%i,end='\r')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_cats[i], test_size=0.1, random_state=13)
    
    bstRF= RandomForestClassifier(n_estimators=200,max_features=40)
    
    bstRF = GridSearchCV(randomForestModel, parameters,verbose=0)
    
    bstRF.fit(X_train,y_train)
    
    print("For the problem of classifying into %d bins the best parameters are"%i)
    print(bstRF.best_params_)
    
    scores_RandomForest.append(bstRF.score(X_test,y_test))
    
    #We save the model for future use.
    pickle.dump(bstRF, open('./clf/videoGames_RF_'+str(i), 'wb'))
{% endhighlight %}

We skip running this block since it would take too much time on my laptop and I have better uses for it. The result for the Random Forest without tunning are as follows 


| Number of bins|  Accuracy for all product | Accuracy for products with more than 10 reviews | Random |
|:-------------:|:---------:|:---------:|:------:| 
| 2             | 0.8289738 | 0.8929765 |0.5    |
| 3             | 0.7014587 | 0.8394648 | 0.333 |
| 4             | 0.6073943 | 0.7558528 | 0.25  |
| 5             | 0.5181086 | 0.6789297 | 0.2   |
| 6             | 0.4635311 | 0.6688963 |0.166  |
| 7             | 0.4305835 | 0.6086956 |0.142  |
| 8             | 0.3827967 | 0.5785953 |0.125  |
| 9             | 0.3516096 | 0.5719063 |0.111  |
| 10            | 0.3277162 | 0.5719063 |0.1 |


Which can be better seen in a graph.


<center>
<img src="{{ '/assets/img/The_models_files/The_models_20_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


### Xgboost

In the next step we try XGBoost since it usually gives better results. We show how to optimize Xgboost in the next section, we don't do this for all the classifiers since this is high time/resources consuming.


{% highlight ruby %}
scores_XGB=[]
for i in range(2,11):
    print('Training for %d bins'%i,end='\r')
    X_train, X_test, y_train, y_test = train_test_split(X, y_cats[i], test_size=0.1, random_state=13)
    xgbModel = XGBClassifier()
    xgbModel.get_xgb_params()['num_class']=i
    xgbModel.get_xgb_params()['objective']="multi:softmax"
    xgbModel.get_xgb_params()['learning_rate'] =0.1
    xgbModel.get_xgb_params()['n_estimators']=140
    xgbModel.get_xgb_params()['max_depth']=5
    xgbModel.fit(X_train,y_train,)
    scores_XGB.append(accuracy_score(y_test,xgbModel.predict(X_test)))
    
    pickle.dump(xgbModel, open('./clf/videoGames_XGB_'+str(i), 'wb'))
{% endhighlight %}

Which gives the results described in the following graph.


<center>
<img src="{{ '/assets/img/The_models_files/The_models_27_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


Maybe more interesting if we put all the model together, note that we definitely get a much better prediction for products that have more reviews. But even in the case of a low number of reviews we see that the reviews are indeed an indicative of the product performance.


<center>
<img src="{{ '/assets/img/The_models_files/The_models_29_0.png' | prepend: site.baseurl }}" alt=""> 
</center>


Another thing we notice, is that the XGBoost is not much better than the random Forest, maybe because we haven't tune the hyperparameters.  

# Hyperparameter tunning.

In this section we show how to optimize the hyperparameters for the xgboost model. We choose the smaller dataset with 10 bins, and separate into training and testing.


{% highlight ruby %}
y=y_cats_10[10]
y=[10 if val[:2]=='10' else int(val[0]) for val in y]
X_train, X_test, y_train, y_test = train_test_split(X_10, y, test_size=0.1, random_state=13)
{% endhighlight %}

The next step is to realize the order of magnitude of estimators that we need, to do this we first create a XGBoost model

{% highlight ruby %}
xgbModel = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= "multi:softmax",
 nthread=4,
 scale_pos_weight=1,
 seed=13)
xgbModel.get_xgb_params()['num_class']=10
{% endhighlight %}


and use cross validation to find when then number of estimator begin to be not make change in our accuracy.

{% highlight ruby %}
xgb_param = xgbModel.get_xgb_params()
xgtrain = xgb.DMatrix(X_train, label=y_train)
xgb_param['num_class'] = 11
cvresult = xgb.cv(
    xgb_param, 
    xgtrain, 
    num_boost_round=xgbModel.get_params()['n_estimators'], 
    nfold=5,
    early_stopping_rounds=50)
{% endhighlight %}

We obtain that we need about 102 estimators. So we go with this in order to find the other parameters. We first go with the max_depth and min_child_weight.


{% highlight ruby %}
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=102, max_depth=5,
 gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= "multi:softmax", nthread=4, scale_pos_weight=1, seed=13), 
param_grid = param_test1,n_jobs=4,iid=False, cv=5)

gsearch1.fit(X_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
{% endhighlight %}

    ([mean: 0.54573, std: 0.02161, params: {'min_child_weight': 1, 'max_depth': 3},
      mean: 0.54203, std: 0.02513, params: {'min_child_weight': 3, 'max_depth': 3},
      mean: 0.55095, std: 0.01680, params: {'min_child_weight': 5, 'max_depth': 3},
      mean: 0.54201, std: 0.01478, params: {'min_child_weight': 1, 'max_depth': 5},
      mean: 0.54574, std: 0.01066, params: {'min_child_weight': 3, 'max_depth': 5},
      mean: 0.54648, std: 0.01484, params: {'min_child_weight': 5, 'max_depth': 5},
      mean: 0.55169, std: 0.01954, params: {'min_child_weight': 1, 'max_depth': 7},
      mean: 0.54761, std: 0.01079, params: {'min_child_weight': 3, 'max_depth': 7},
      mean: 0.54275, std: 0.00697, params: {'min_child_weight': 5, 'max_depth': 7},
      mean: 0.54909, std: 0.01307, params: {'min_child_weight': 1, 'max_depth': 9},
      mean: 0.54536, std: 0.00898, params: {'min_child_weight': 3, 'max_depth': 9},
      mean: 0.54389, std: 0.00756, params: {'min_child_weight': 5, 'max_depth': 9}],
     {'max_depth': 7, 'min_child_weight': 1},
     0.55169490036143287)



Next we cross validate for gamma 


{% highlight ruby %}
param_test2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=110, max_depth=7,
 min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
 objective= "multi:softmax", nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2,n_jobs=4,iid=False, cv=5)
gsearch2.fit(X_train,y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
{% endhighlight %}

    ([mean: 0.54836, std: 0.01822, params: {'gamma': 0.0},
      mean: 0.54910, std: 0.01418, params: {'gamma': 0.1},
      mean: 0.55655, std: 0.01242, params: {'gamma': 0.2},
      mean: 0.54872, std: 0.01457, params: {'gamma': 0.3},
      mean: 0.54426, std: 0.00920, params: {'gamma': 0.4}],
     {'gamma': 0.2},
     0.55654553708434285)



follow by the subsample size and the colsample_bytree


{% highlight ruby %}
param_test3 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=110, max_depth=7,
 min_child_weight=1, subsample=0.8, gamma=0.2, colsample_bytree=0.8,
 objective= "multi:softmax", nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3,n_jobs=4,iid=False, cv=5)
gsearch3.fit(X_train,y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
{% endhighlight %}

    ([mean: 0.53828, std: 0.00689, params: {'subsample': 0.6, 'colsample_bytree': 0.6},
      mean: 0.54908, std: 0.01704, params: {'subsample': 0.7, 'colsample_bytree': 0.6},
      mean: 0.54127, std: 0.01434, params: {'subsample': 0.8, 'colsample_bytree': 0.6},
      mean: 0.53792, std: 0.01848, params: {'subsample': 0.9, 'colsample_bytree': 0.6},
      mean: 0.54127, std: 0.01538, params: {'subsample': 0.6, 'colsample_bytree': 0.7},
      mean: 0.54874, std: 0.01445, params: {'subsample': 0.7, 'colsample_bytree': 0.7},
      mean: 0.54835, std: 0.00875, params: {'subsample': 0.8, 'colsample_bytree': 0.7},
      mean: 0.54202, std: 0.01154, params: {'subsample': 0.9, 'colsample_bytree': 0.7},
      mean: 0.54539, std: 0.01139, params: {'subsample': 0.6, 'colsample_bytree': 0.8},
      mean: 0.54650, std: 0.01588, params: {'subsample': 0.7, 'colsample_bytree': 0.8},
      mean: 0.55655, std: 0.01242, params: {'subsample': 0.8, 'colsample_bytree': 0.8},
      mean: 0.54276, std: 0.01903, params: {'subsample': 0.9, 'colsample_bytree': 0.8},
      mean: 0.54646, std: 0.01259, params: {'subsample': 0.6, 'colsample_bytree': 0.9},
      mean: 0.54685, std: 0.01228, params: {'subsample': 0.7, 'colsample_bytree': 0.9},
      mean: 0.54462, std: 0.01579, params: {'subsample': 0.8, 'colsample_bytree': 0.9},
      mean: 0.54203, std: 0.01190, params: {'subsample': 0.9, 'colsample_bytree': 0.9}],
     {'colsample_bytree': 0.8, 'subsample': 0.8},
     0.55654553708434285)



So, it seems that the ones we started with were already a good choice. Finally, we do the reg_alpha


{% highlight ruby %}
param_test4 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=110, max_depth=7,
 min_child_weight=1, subsample=0.8, gamma=0.2, colsample_bytree=0.8,
 objective= "multi:softmax", nthread=4, scale_pos_weight=1,seed=13), 
 param_grid = param_test4,n_jobs=4,iid=False, cv=5)
gsearch4.fit(X_train,y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

{% endhighlight %}

    ([mean: 0.54461, std: 0.01104, params: {'reg_alpha': 1e-05},
      mean: 0.54272, std: 0.01684, params: {'reg_alpha': 0.01},
      mean: 0.54313, std: 0.01283, params: {'reg_alpha': 0.1},
      mean: 0.54870, std: 0.01768, params: {'reg_alpha': 1},
      mean: 0.44503, std: 0.01289, params: {'reg_alpha': 100}],
     {'reg_alpha': 1},
     0.54870361819532376)



As the last step we decrease the learning rate and increase the number of estimators to get our classifier.


{% highlight ruby %}
bstXGB= XGBClassifier(
    learning_rate =0.01, 
    n_estimators=1000, 
    max_depth=7,
    min_child_weight=1, 
    subsample=0.8, 
    gamma=0.2, 
    colsample_bytree=0.8,
    objective= "multi:softmax", 
    nthread=4, 
    scale_pos_weight=1,
    seed=13,
    reg_alpha=1)
bstXGB.fit(X_train,y_train)
{% endhighlight %}

and use the testing data to see note that we get an accuracy of 0.57525083612040129 which compared with the score of 0.53846153846153844 we got before, it is a quite decent improvement. (Note that this is about the same we got with Random Forest).


# Final Notes

The reviews dataset contains important information about the salesRank, as such it can be used to give a prediction. Even, if the prediction is far from perfect we have shown that it can be used as a feature to improve our understanding of the products. 

We also note that there are many different ways to improve the results, there are ranking ML techniques that may give better results (at the cost of longer training time), or we could mine the original data to get more features. From the pictures, from recommenders, from the similar products, etc. 

In order to implement this in production, there are several ways to improve the code, which would turn with a higher performance for large datasets. Fortunately for us, the only feature that required heavy processing is the sentiment analysis, which can be performed offline. All the other techniques we used have a fast performance for larger datasets. 


{% highlight ruby %}

{% endhighlight %}
