#!/usr/bin/env python
# coding: utf-8

# In[117]:


from IPython.display import Image


# In[60]:


import pandas as pd
import numpy as np
from wordcloud import STOPWORDS
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")


# In[65]:


df1=pd.read_csv("dreaddit-train.csv")


# In[66]:


df3=pd.read_csv("dreaddit-test.csv")


# In[67]:


df1.shape


# In[68]:


df3.shape


# In[69]:


df1.sample()


# In[70]:


df3.sample()


# In[71]:


# We merged the two files. We have completed the missing data.
df=df1.append(df3) 


# In[72]:


df.shape


# In[73]:


df.head()


# In[74]:


df.columns


# In[75]:


df.info()


# In[76]:


df.isnull().sum()


# In[77]:


from textblob import TextBlob


# In[78]:


TextBlob("the best").polarity #We find the positive or negative of the words.


# In[79]:


TextBlob("the best").sentiment


# In[80]:


def detect_sentiment(text):
    return TextBlob(text).sentiment.polarity


# In[81]:


df2=df[["text"]]


# In[82]:


df2.head()


# In[83]:


df2["sentiment"]=df2["text"].apply(detect_sentiment)


# In[84]:


df2.head()


# In[85]:


df2.sentiment.value_counts()


# In[86]:


import nltk
import re
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string


# In[87]:


stopwords = set(stopwords.words("english"))


# In[88]:


#we clean up unnecessary marks
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopwords]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
df2["text"] = df2["text"].apply(clean)


# In[89]:


df2["text"]


# In[90]:


def wc(data,bgcolor):
    plt.figure(figsize=(20,20))
    mask=np.array(Image.open('Stress.png'))
    wc=WordCloud(background_color=bgcolor,stopwords=STOPWORDS,mask=mask)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis("off")


# In[83]:


wc(df2.text,'white')


# In[91]:


df2["label"]=df["label"].map({0: "No Stress", 1: "Stress"})
df2=df2[["text", "label"]]


# In[92]:


df2.head()


# In[93]:


df2["sentiment"]=df2["text"].apply(detect_sentiment)


# In[94]:


df2.head()


# In[95]:


import seaborn as sns


# In[96]:


sns.countplot(x=df2.label)


# In[97]:


x=df2.text
y=df2.label


# In[98]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# In[99]:


vect=CountVectorizer(stop_words="english")


# In[100]:


x=vect.fit_transform(x)


# In[101]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)


# In[102]:


mb=MultinomialNB()


# In[103]:


tahmin=mb.fit(x_train,y_train).predict(x_test)


# In[104]:


accuracy_score(tahmin,y_test)


# In[105]:


from sklearn.tree import DecisionTreeClassifier


# In[106]:


d=DecisionTreeClassifier()


# In[107]:


d.fit(x_train,y_train)


# In[108]:


tahmin1=d.predict(x_test)


# In[109]:


accuracy_score(y_test,tahmin1)


# In[115]:


user="Sometime I feel like I need some help"


# In[116]:


df2=vect.transform([user]).toarray()
output=d.predict(df2)
print(output)


# In[ ]:




