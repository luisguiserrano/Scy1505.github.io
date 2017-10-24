---
layout: post
title:  "Intro to Attention Mechanisms"
date:   2017-10-24
category: general
---
In this blog post we attempt to describe attention mechanisms. This is motivated by the fact that I haven't found any good reference for its implementation; most of the examples online are either wrong or unnecesarilly complex, so I decided to create my own. 

The outline is as follows.

- We explain attention mechanisms.
- We build a simple mechanism.
- We motivate state of the art. 

Something interesting that came out of it, was a beautiful proof of the character embedding and how it helps for the problem. 

# Attention Mechanisms

Attention mechanisms were build with the purpose of helping other ML/DL tehcniques focus their effort in certain regions that contain more relevant information for the tast at hand. I believe they were first introduced for working with images but it was later used in NLP. 

In NLP it is typical to have a problem where a sequence (word,phrase,etc) needs to be converted into some other sequence. For example translation. 

<div align="center">
<img src="{{ '/assets/img/intro_to_attention/translations.png' | prepend: site.baseurl }}" alt="" width="360" > 
</div>


We need to know which words in the input sequence correspond to which words in the output. So it will be good create a tool that help us do that. 

Let's make this more formal meanwhile keeping an example in mind. Suppose that we want to create something that takes a word with a typo and outputs the correct word. 

Int the formal side we have a sequene $$ a_1 \rightarrow a_2 \rightarrow \cdots \rightarrow a_n $$ which we want to transform into $$ b_1 \rightarrow b_2 \rightarrow \cdots \rightarrow b_m$$

For example 

<div align="center">
<img src="{{ '/assets/img/intro_to_attention/wword2word.png' | prepend: site.baseurl }}" alt="" width="600" > 
</div>


This is clearly a sequence to sequence problem and we first can do a character embedding and in the output layer we can use a softmax to predict the right character

<div align="center">
<img src="{{ '/assets/img/intro_to_attention/emb2soft.png' | prepend: site.baseurl }}" alt="" width="600" > 
</div>


Now every sequence to sequence is usually obtained as an encodding part and a decoding part which are usually build out of RNN.

<div align="center">
<img src="{{ '/assets/img/intro_to_attention/decoder.png' | prepend: site.baseurl }}" alt="" width="480" > 
</div>

For our example we will make the cells a 3 layer stack of LSTM's. Let's summarize this slighly as 

<div align="center">
<img src="{{ '/assets/img/intro_to_attention/seq2seq.png' | prepend: site.baseurl }}" alt="" width="480" > 
</div>

The things to remeber here is that the horizontal (pink) arrows are states, each state encodes the sequence that it has left behind. Now we would like for the states on the decoder to have a push towards finding the right prediction, the problem is that it may have too much info from before. Then the idea is to compare the current decoder state with all the states in the encoder

<div align="center">
<img src="{{ '/assets/img/intro_to_attention/attention.png' | prepend: site.baseurl }}" alt="" width="480" > 
</div>

Let's simplify the problem and only try to find out the character with the highest probability of been a typo.


<div align="center">
<img src="{{ '/assets/img/intro_to_attention/typoAt.png' | prepend: site.baseurl }}" alt="" width="600" > 
</div>

Let's now implement this ideas. We encourage the reader to look at the [python code in the github](https://github.com/Scy1505/wrongChar), but we only show the network construction here. First the encoder which is is just a simple RNN with lstm cells and number of layers given by nb_layer=3.


{% highlight ruby %}
with tf.name_scope("encoder"):

	with tf.variable_scope('encoder_'):

		lstms=[tf.contrib.rnn.LSTMCell(hidden_dim) for i in range(nb_layers)]

		encoder = tf.contrib.rnn.MultiRNNCell(lstms)

		initial_state=encoder.zero_state(batch_size,dtype=tf.float32)

		output,transition_state= tf.nn.dynamic_rnn(encoder,output,initial_state=initial_state,dtype=tf.float32)

{% endhighlight %}

The attention mechanism that we use is a simple one (and sort of a cheat) where instead of taking the states going from one cell to the next we take the (weighted) output of the LSTM layers. This will play the role of a context vector. 


{% highlight ruby %}
with tf.name_scope("attention"):

	attention_matrix = tf.Variable(
            	tf.random_normal(shape=(longest_seq*hidden_dim,1)),name="attention_matrix")
	attention_bias=tf.Variable(
            	tf.zeros(1),name="atention_bias")

	decoder_input = tf.matmul(tf.reshape(output, (-1,longest_seq*hidden_dim)),attention_matrix)+attention_bias
	decoder_input = tf.reshape(decoder_input,shape=(batch_size,1,1))

{% endhighlight %}

Finally, for the decoder we use a simple (lenght one) LSTM RNN


{% highlight ruby %}
with tf.name_scope("decoder"):
	with tf.variable_scope('decoder_'):
            

		lstms_decoder=[tf.contrib.rnn.LSTMCell(hidden_dim) for i in range(nb_layers)]

		decoder = tf.contrib.rnn.MultiRNNCell(lstms_decoder)

		output_decoder,_= tf.nn.dynamic_rnn(decoder,decoder_input,initial_state= transition_state, dtype=tf.float32)
{% endhighlight %}

As result comes, a small script allows us to represent the attention 

<div align="center">
<img src="{{ '/assets/img/intro_to_attention/results.png' | prepend: site.baseurl }}" alt="" width="240" > 
</div>

Let's study this results a bit, note that shorter words are more difficult to deal with. The reason is that the solution may not be unique. (Think where's the typo in pake? is it at p (cake)?, is it at a (poke)?, or is it at k (pate)?. For longer words it does an amazing job. More interesting is that for vowels it also does a great job. Why, this is thanks to the character embedding that is learning. (which in theory is one of the best embeddings for the task a hand). Here is how it looks. 

<div align="center">
<img src="{{ '/assets/img/intro_to_attention/testing5.gif' | prepend: site.baseurl }}" alt="" width="600" > 
</div>

Note how it separetes the vowels, and it separates the most common letter in the english language. 

# Where to go from here.

If you want to implement a more advanced attention mechanism you should try using [tensor2tensor](https://github.com/tensorflow/tensor2tensor) or the attention mechanism prebuild in Tensorflow, just keep in mind that at the moment of this writting there are some versioning problem and the documentation is kind of terrible. 
