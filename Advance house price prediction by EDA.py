#!/usr/bin/env python
# coding: utf-8

# ## Advance house price prediction by EDA
# 

# In[108]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns  

pd.pandas.set_option('display.max_columns',None)


# In[109]:


dataset=pd.read_csv('train.csv')

print(dataset.shape)


# In[110]:


ds.head()


# In[111]:


features_with_na=[features for features in dataset.columns if dataset[features].isnull().sum()>1]

for feature in features_with_na:
    print(feature,np.round(dataset[feature].isnull().mean(),4),'%missing values')


# In[112]:


for feature in features_with_na:
    data=dataset.copy()
    
    data[feature]=np.where(data[feature].isnull(),1,0)
    
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()


# In[113]:


print("Id of house{}".format(len(dataset.Id)))


# In[114]:


numerical_features=[feature for feature in dataset.columns if dataset[feature].dtypes!='O']

print('Number of numerical variables:',len(numerical_features))
dataset[numerical_features].head()


# In[115]:


year_feature=[feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
year_feature


# In[116]:


for feature in year_feature:
    print(feature,dataset[feature].unique())


# In[117]:


dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('YearSold')
plt.ylabel('Median House price')
plt.title('House Price vs Year sold')


# In[118]:


year_feature


# In[119]:


for feature in year_feature:
    if feature !='YrSold':
        data=dataset.copy()
        data[feature]=data['YrSold']-data[feature]
        
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# In[120]:


discrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature+['Id']]

print("discrete varaiables count: {}".format(len(discrete_feature)))


# In[121]:


discrete_feature


# In[122]:


dataset[discrete_feature].head()


# In[123]:


for feature in discrete_feature:
    data=dataset.copy()
    sns.barplot(x=feature,y='SalePrice', data=data, ci=False, estimator=np.median)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# In[124]:


continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]

print("continuous_feature count:{}".format(len(continuous_feature)))                   


# In[125]:


for feature in continuous_feature:
    data=dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("count")
    plt.title(feature)
    plt.show()


# In[126]:



for feature in continuous_feature:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title(feature)
        plt.show()


# In[127]:


for feature in continuous_feature:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# In[128]:


categorical_feature=[feature for feature in dataset.columns if dataset[feature].dtypes=='O']

categorical_feature


# In[129]:


dataset[categorical_feature].head()


# In[130]:


for feature in categorical_feature:
    print('The feature is {} and number of categories are {} '.format(feature,len(dataset[feature].unique())))


# In[131]:


for feature in categorical_feature:
    data=dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# ## Feature Engineering

# In[132]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

pd.pandas.set_option('display.max_columns',None)


# In[133]:


dataset=pd.read_csv('train.csv')

dataset.head()


# In[134]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(dataset,dataset['SalePrice'],test_size=0.1,random_state=0)


# In[135]:


X_train.shape,X_test.shape


# In[136]:


feature_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes=='O']

for feature in feature_nan:
    print("{} : {}%missing values".format(feature,np.round(dataset[feature].isnull().mean(),4)))


# In[137]:


def replace_cat_feature(dataset,feature_nan):
    data=dataset.copy()
    data[feature_nan]=data[feature_nan].fillna('missing')
    
    return data
dataset=replace_cat_feature(dataset,feature_nan)
dataset[feature_nan].isnull().sum()


# In[138]:


dataset.head()


# In[139]:


numerical_with_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes!='O']

for feature in numerical_with_nan:
    print("{}: {}%missing value".format(feature,np.round(dataset[feature].isnull().mean(),4)))


# In[140]:


for feature in numerical_with_nan:
    median_value=dataset[feature].median()
    
    dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)
    dataset[feature].fillna(median_value,inplace=True)
    
    dataset[numerical_with_nan].isnull().sum()
    
    


# In[141]:


dataset.head()


# In[142]:


dataset[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()


# In[143]:


year_feature


# In[144]:


for feature in year_feature:
    if feature!='yrSold':
        data=dataset.copy()
        data[feature]=data['YrSold']-data[feature]
        
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# In[145]:


data.head()


# In[146]:


import numpy as np
num_feature=['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice']

for feature in num_feature:
    dataset[feature]=np.log(dataset[feature])


# In[152]:


dataset.head(50)


# In[148]:


categorical_feature=[feature for feature in dataset if dataset[feature].dtype!='O']

categorical_feature


# In[149]:


for feature in categorical_feature:
    temp=dataset.groupby(feature)['SalePrice'].count()/len(dataset)
    temp_df=temp[temp>0.01].index
    dataset[feature]=np.where(dataset[feature].isin(temp_df),dataset[feature],'Rare_var')


# In[153]:


dataset.head(50)

