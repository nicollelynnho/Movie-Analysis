#!/usr/bin/env python
# coding: utf-8

# Movie Database Analysis by Nicolle Ho

# In[1]:


#Ask Questions


# This report analyzes the tmbd-movies database. The purpose of the analysis is to gain insight into movie popularity. What are the attributes that are associated with movies that are popular?

# In[2]:


#Wrangle Data


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


# In[4]:


#Read Database
df = pd.read_csv("tmdb-movies.csv")
df.head()
df.shape


# In[5]:


#Check for missing values
df.isnull()
df.isnull().sum()
#below there are missing values for several fields 


# In[6]:


#Dont need to drop null items because the fields we are analyzing do not have nulls
#df = df.dropna()
#df.isnull().sum()


# In[7]:


#check for duplicate rows 
df = df.drop_duplicates(keep = 'first')
#values return false so it is not a duplicate row


# In[8]:


#check for unique responses to see if there are hidden trues
df.duplicated(keep='first').nunique()
#only one unique response, so all values are False (not duplicated)


# In[9]:


#check data types
df.dtypes
#the object data types have many values separated by | so this requires more cleaning


# In[10]:


#Determine distribution of popularity and check for outliers
print(df['popularity'].describe())
pop = df['popularity']
n, bins, patches = plt.hist(x=pop, bins=40)
plt.title('Popularity Distribution')
plt.xlabel('Popularity Score')
plt.ylabel('Count')


# This graph shows how popularity scores are distributed across 10866 movies. The popularity scores are skewed right, so there are fewer movies that are popular. This is intuitive. We can also see that there are several outliers in the upperbound. There are several movies that are much more popular than the rest of the movies. 

# In[11]:


#Boxplot to view distribution and check for outliers
fig1, ax1 = plt.subplots()
ax1.set_title('Popularity Boxplot')
ax1.boxplot(pop, vert = False, notch = True)
ax1.set_xlabel('Popularity Score')
#Another view to show that popularity is skewed right
#Popularity scores are concentrated on the lower end


# This boxplot is another view of how popularity is distributed across all the movies. This view is interesting because we can see where are the quartiles. The first three quartiles are very concentrated in the lower popularity score region. The last quartile has the largest range. This confirms that the majority of movies are not popular. There is high variability in popularity for the popular movies. Because of these large outliers and variation, both the high and the low outliers will be excluded from the rest of the analysis. 

# In[13]:


#subset of the inner 50% to remove outliers
df2 = df.loc[(df.popularity > 0.384079) & (df.popularity < 1.538639)]
df2.head()


# In[14]:


#Determine popularity distribution of inner 50%
print(df2['popularity'].describe())
pop2 = df2['popularity']
n, bins, patches = plt.hist(x=pop2, bins=40)
plt.title('Popularity Distribution')
plt.xlabel('Popularity Score')
plt.ylabel('Count')
#There are fewer movies that are highly popular
#this is intuitive because most movies are not popular


# When examining the inner 50%, we can see that there is negative linear relationship between popularity score and count. The more popular a movie is, fewer movies exist with that level of popularity. 

# In[16]:


#Boxplot to view distribution and check for outliers
print(df2['popularity'].describe())
fig1, ax1 = plt.subplots()
ax1.set_title('Popularity Boxplot')
ax1.set_xlabel('Popularity Score')
ax1.boxplot(pop2, vert = False, notch = True)
#There are fewer movies that are highly popular
#this is intuitive because most movies are not popular


# This is a boxplot view of the inner 50%. The median popularity score is 0.63. Each quartile increases in size, which confirms that popularity is skewed right. 

# In[17]:


#Exploratory Data Analysis


# In[19]:


#Remove erroneous zero budget / revenue values
df2 = df[(df.budget_adj != 0) & (df.revenue_adj != 0)]


# In[21]:


#Split the data into quintiles by popularity to see how groups behave on average
df2['quantile'] = pd.qcut(df2['popularity'],5,labels=["Q1","Q2","Q3","Q4","Q5"])
df2.head()


# In[23]:


#confirm that errous zero budget / revenue movies were removed 
df2 = df2.sort_values(by='budget_adj')
df2.head()


# In[24]:


#Plot popularity vs budget non outlier movies
plt.title("Popularity vs Budget for middle 50%")
plt.xlabel("Popularity Score")
plt.ylabel("Adjusted Budget in dollars")
plt.scatter(df2['popularity'], df2['budget_adj'], s=80, c='b', marker="o")
#Difficult to see a trend between popularity and budget


# We are plotting popularity vs budget to try to determine a relationship. However because of the numerous and cluttered data points it is difficult to see a clear trend. 

# In[25]:


#Calculate average budget and revenue for each quintile
budget = df2.groupby('quantile')['budget_adj'].mean()
revenue = df2.groupby('quantile')['revenue_adj'].mean()


# In[27]:


n_groups = 5

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, budget, bar_width,
                alpha=opacity, color='b', error_kw=error_config,
                label='Adjusted Budget')

rects2 = ax.bar(index + bar_width, revenue, bar_width,
                alpha=opacity, color='r', error_kw=error_config,
                label='Adjusted Revenue')

ax.set_xlabel('Quantile')
ax.set_ylabel('Dollars')
ax.set_title('Adjusted and Budget and Revenue by Quantile')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('Q1', 'Q2', 'Q3', 'Q4', 'Q5'))
ax.legend()

fig.tight_layout()
plt.show()


# We can see trends more clearly when data is split into quantiles. More popular movies have larger budgets than less popular movies. More popular movies generate larger revenues than less popular movies. 

# In[28]:


#Popularity vs Runtime for all movies
plt.title("Popularity vs Runtime for all movies")
plt.xlabel("Popularity Score")
plt.ylabel("Minutes")
plt.scatter(df2['popularity'], df2['runtime'], s=80, c='g', marker="o")
#There are two outliers -- two movies that are really long


# Popularity is plotted against runtime to try and determine a relationship. The trend is unclear because of the concentration of data points on the left hand side of the graph. 

# In[29]:


#Sort the values by runtime to determine outliers which is the 338 min movie
df2 = df2.sort_values(by='runtime')
df2.tail()


# In[30]:


#Drop runtime outliers
df2 = df2[(df2.runtime < 337)]
df2.tail()
#338 runtime values were dropped 


# In[31]:


#Runtime by quintile
runtime = df2.groupby('quantile')['runtime'].mean()

n_groups = 5

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index+bar_width/2, runtime, bar_width,
                alpha=opacity, color='b', error_kw=error_config,
                label='Runtime')

ax.set_xlabel('Quantile')
ax.set_ylabel('Minutes')
ax.set_title('Runtime by Quintile')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('Q1', 'Q2', 'Q3', 'Q4', 'Q5'))
ax.legend()
fig.tight_layout()
plt.show()
runtime


# Breaking the data into quintiles allows us to view the data more clearly. All the bar heights are approximately the same size, so there does not appear to be a relationship between runtime and movie popularity. 

# In[32]:


df2['genres']
df2['genres'] = df2['genres'].str.split('|')
df2.groupby('quantile')['genres'].apply(list)


# In[33]:


df2['genres'].values.reshape(-1)


# In[34]:


df2.shape


# In[ ]:


#Draw Conclusions


# This report analyzes attributes of popular movies. 
# 
# The dataset is rich and sufficient for gaining greater insight into which factors are related to movie popularity. The original dataset had 10866 observations (movies) and 21 fields that include budget, revenue, runtime, genre, etc. After creating a subset of the inner 50% based on popularity and dropping movies with zero budget / revenue, the final analyzed dataset had 3853 observations and 22 fields (added quintile field). The dataset posed additional challenges because fields such as genre and production company had several attributes separated by |. This will require additional data cleaning and analytical reflection as some movies are listed as multiple genres. 
# 
# More popular movies (higher quintiles) have higher average budgets and average revenues than less popular movies movies in (lower quintiles). There is no obvious relationship between runtime and movie popularity. Next steps of analysis include investigating how genres, production companies, acting casts, and directors contribute to movie popularity. 

# In[ ]:




