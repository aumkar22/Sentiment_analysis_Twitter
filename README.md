Introduction
============

Twitter is a social media platform which users access for global news or
for posting their opinions in the form of a text status called as a
‘tweet’. Tweets have a 140 character limit per tweet (280 characters
since 2017). Due to this limit, its easier for Twitter users to voice
their opinions on different topics such as brands, politics, etc. Thus
sentiment analysis on tweets can provide an insight into what people’s
opinion is on these topics.

Besides classification of tweets for getting opinions of users,
sentiment analysis can also be performed to get predictions similar to
the predictions made by Nate Silver, who analyzed millions of tweets and
got the predictions correct for 49 out of the 50 states for the
presidential elections of 2008 and all the 50 correct for the one in
2012 [@silver2012signal].

I developed an Artificial Neural Network (ANN) model based on Term
Frequency-Inverse Document Frequency (TF-IDF) weighted Word2Vec model
[@mikolov2013efficient] to build a sentiment classifier and achieved an
accuracy of $\sim86\%$ on the test set. Thus the research question for
this project is *‘By performing sentiment analysis, is it possible to
predict whether a tweet is positive or negative?’*

Review of related work
======================

There are many approaches and models developed for the task at hand. One
of the most notable one is by [@Go_Bhayani_Huang_2009] at the Stanford
University. Their approach uses emoticons
[@Go_Bhayani_Huang_2009][@read2005using] as labels as they are easy to
express opinions, for instance ‘:)’ indicates a positive emotion while a
‘:(’ indicates negative emotion. These labels are noisy as its not
possible to detect sarcasm or mismatched emotions. They train Naive
Bayes, Maximum Entropy and Support Vector Machines (SVM) models on a
data set consisting of 1.6 million tweets and test it on a manually
collected data of 359 tweets and achieve an accuracy of above 80%.
Similar approach has been taken by [@pak2010twitter].

Another approach uses tweet hashtags (\#) for sentiment classification
[@wang2011topic]. For this, they constructed a Hashtag Graph Model where
the nodes are hashtags and the edges between two nodes exist if the two
hashtags co-occur in the same tweet. To assign sentiments to each
hashtag, they use three algorithms: Loopy Belief Propagation (LBP),
Relaxation Labeling (RL) and Iterative Classification Algorithm (ICA).
They build this hashtag-level classifier on top of tweet-level sentiment
classification results for which they used two-stage SVM model. Using
this model, they achieve an accuracy of $\sim77\%$.

One more interesting approach includes learning sentiment-specific word
embedding (SSWE) for sentiment classification [@tang2014learning]. They
develop three neural networks to learn the word embedding and train the
model on a data set of 10 million tweets. They achieve an accuracy of
$\sim86\%$.

Data and Pre-processing steps
=============================

The data set used to perform the sentiment analysis was a labeled,
fairly balanced data set of 1.6 million tweets provided publicly and
mentioned in [@Go_Bhayani_Huang_2009]. The data consisted of six fields:
The polarity of the tweet, with 0 for negative and 4 for positive. A
unique tweet ID, the date of the tweet, the query, ID of the user who
tweeted and the text of the tweet.

The first step of pre-processing performed was changing the polarity of
the tweets from 0 and 4 to 0 and 1 for negative and positive
respectively. The tweets were then converted into tokens. Later user
mentions (@), hashtags (\#) and URLs were removed as they do not provide
a lot of semantic information.

Approach
========

Initially, the data was split in train and test set with the ration
80:20. A baseline SVM model was trained and then tested on the test set.
Similar process was followed for the main model. Lastly, K-fold cross
validation was performed on the entire data set.

Baseline
--------

Support Vector Machine is a well known classification model used
extensively in the machine learning field. For this project, a baseline
SVM model was trained using a linear kernel. To overcome scalability
issue, a Min-Max Scalar was used to scale the features in the range of 0
to 1. Also, as SVMs do not scale to larger data sets, the model was
trained in batches with a batch size set to 10000. The model achieved an
accuracy of $\sim76\%$ on the test set after this process.

Word2Vec Model
--------------

The word embedding model proposed in [@mikolov2013efficient] deals with
converting words into vectors which can be represented in a huge
multi-dimensional vector space. The model proposed consists of two
algorithms. First, Continuous Bag of Words (CBOW) which predicts current
word from surrounding context words from a window of a specific size.
The second, Skip Gram model which predicts surrounding context words
given the current word. From the weights of the trained model then a
vector is generated which ‘holds’ the meaning of the word. In the vector
space, cosine distance is measured and used to answer questions such as
*‘What is x for*
`$x = vector(`biggest') - vector(`big') + vector(`small')?$’. Thus, words
which are semantically closer, appear closer in the vector space than
words which have more different meaning. The original model is a feed
forward two layer neural network which processes text. The output of the
network is a vocabulary with a vector attached to each item.

For this project, post pre-processing generated tokens are fed into the
word2vec model to generate vectors in a vector space of dimension 500. A
threshold of value 10 was applied to filter out less occurring words.
The model is then trained to update the weights. Figure
\[fig:word\_vecs\] shows a sample of generated vectors. As we can see,
words such as ‘car’ and ‘bike’; ‘weekend’, ‘time’ and ‘day’ form a
cluster in the vector space.

![Word vectors representation[]{data-label="fig:word_vecs"}](word_vecs.png)

Sentiment Classifier
--------------------

For the classification purpose, a two layer neural network was used. In
order to classify the tweets, the word embeddings were converted into
Term Frequency-Inverse Document Frequency (TF-IDF) weight vectors.
TF-IDF technique penalizes common words by assigning them low weights
and assigns high weights to important words with respect to the corpus.
Hence, weighted average tweet vector was created, given the list of
tokens as an input.

Binary cross entropy loss function was used along with sigmoid
activation function was used in the last layer to predict the true
(‘positive’) class.

Results and Evaluation
======================

The results achieved using the three methods is shown in the figure
\[fig:result\]

![ROC curve[]{data-label="fig:result"}](Results.png)

The main model performed the best on the test set with an accuracy of
$\sim86\%$. Further, Receiver operating characteristic (ROC) curve plot
was made to get an estimate of false positive rate. The Area under the
curve (AUC) was calculated to be 0.86877 as seen in the figure
\[fig:roc\].

For more evaluation purposes, K-fold cross validation was used on the
entire data set with K=5. The mean accuracy achieved from cross
validation was $\sim79\%$ with a mean standard deviation of $\sim0.09\%$
over each fold.

Both the main model and the cross validated model performed better than
the baseline SVM model which achieved an accuracy of $\sim76\%$ on the
test set.

![ROC curve[]{data-label="fig:roc"}](ROC.png)

Discussion and Conclusion
=========================

The model gives a high accuracy of $\sim86\%$ on the test set which is
equivalent to the accuracy achieved in
[@Go_Bhayani_Huang_2009][@tang2014learning]. The model performs well
because of two main reasons. First, the test data has similar
characteristics as the train data. As mentioned earlier, the data is
labeled on the basis of emoticons which are present in the entire data
set. Second, word2vec performs really good to determine semantic
relations when trained on a large corpus, in this case $\sim1.3$ million
tweets. It would be interesting to check the performance of the model on
data with different characteristics such as tweets without emoticons and
also on tweets about a certain topic or a hashtag.
