# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 12:50:08 2018

@author: Aumkar
"""

import nltk
import tweepy
from nltk.corpus import stopwords
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

#consumer key, consumer secret, access token, access secret.

ckey="t7mlx9A9aFHrOR91SvmmHI8I4"
csecret="4HQFjD4p921xcI2AZQ2mnqxXkI4GnmllkaldXITl8XPV8hy4fb"
atoken="927974990607208449-KcVvOqRaXvFqgmsMoeoheCRw0LhZm71"
asecret="IgR6Q5tZxbdbvbMIL9WzgAv6eSV2uvlKuLXICJe9XmGFG"

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
 
api = tweepy.API(auth)