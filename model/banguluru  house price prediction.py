#!/usr/bin/env python
# coding: utf-8

# # tut1

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)


# In[2]:


df1=pd.read_csv("bengaluru_house_prices.csv")
df1.head()


# In[3]:


df1.shape


# In[4]:


df1.columns


# In[5]:


df1.groupby("area_type")["area_type"].agg("count")


# In[6]:


df2=df1.drop(["area_type","society","availability","balcony"],axis="columns")
df2.head()


# In[7]:


df2.isnull().sum()


# In[8]:


df3=df2.dropna()
df3.head()


# In[9]:


df3.isnull().sum()


# In[10]:


df3["size"].unique()


# In[11]:


df3["bhk"]=df3["size"].apply(lambda x:int(x.split(" ")[0]))
df3.head()


# In[12]:


df3.bhk.unique()


# In[13]:


df3[df3.bhk>=20]


# In[14]:


df3["total_sqft"].unique() 


# In[15]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[16]:


df3[~df3.total_sqft.apply(is_float)].head()


# In[17]:


def conv_sqft_to_num(x):
    token=x.split("-")
    if(len(token)==2):
        return (float(token[0])+float(token[1]))/2.0
    try:
        return float(x)
    except:
        return None


# In[18]:


conv_sqft_to_num('23345')


# In[19]:


conv_sqft_to_num('4-10')


# In[20]:


conv_sqft_to_num("34gjk")


# In[21]:


df4=df3.copy()
df4["total_sqft"]=df4["total_sqft"].apply(conv_sqft_to_num)
df4.head(3)


# In[22]:


df4.loc[30]


# # tut2

# In[23]:


df4.head(3)


# In[24]:


df5=df4.copy()


# In[25]:


df5.isnull().sum()


# In[26]:


df5['price_per_sqft']=df5['price']*100000/df5["total_sqft"]
df5.head()


# In[27]:


df5.isnull().sum()


# In[28]:


len(df5.location.unique())


# In[29]:


df5.location=df5.location.apply(lambda x:x.strip())
location_states=df5.groupby(["location"])["location"].agg("count").sort_values(ascending=False)
location_states


# In[30]:


len(location_states[location_states<=10])


# In[31]:


location_states_less_than_10=location_states[location_states<=10]
location_states_less_than_10


# In[32]:


len(df5.location.unique())


# In[33]:


df5.location=df5.location.apply(lambda x: 'other' if x in location_states_less_than_10 else x)
len(df5.location.unique())


# In[34]:


df5.head(10)


# # tut3

# In[35]:


df5[df5.total_sqft/df5.bhk<300].head()


# In[36]:


df5.shape


# In[37]:


df6=df5[~(df5.total_sqft/df5.bhk<300)]
df6.head()


# In[38]:


df6.shape


# In[39]:


df6.price_per_sqft.describe()


# In[40]:


def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby("location"):
        m=np.mean(subdf.price_per_sqft)
        sd=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-sd)) & (subdf.price_per_sqft<=(m+sd))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7=remove_pps_outliers(df6)
df7.shape


# In[41]:


def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location) &(df.bhk==2)]
    bhk3=df[(df.location==location) &(df.bhk==3)]
    matplotlib.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='Blue',label="2 bhk",s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,color='Red',marker='+',label="3 bhk",s=50)
    plt.xlabel("Total square feet")
    plt.ylabel("Price")
    plt.title("Location")
    plt.legend()
plot_scatter_chart(df7,"Hebbal")


# In[42]:


def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby("location"):
        bhk_states={}
        for bhk, bhk_df in location_df.groupby("bhk"):
            bhk_states[bhk]={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std':np.std(bhk_df.price_per_sqft),
                'count':bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby("bhk"):
            state=bhk_states.get(bhk-1)
            if state and state['count']>5:
                exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(state['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8=remove_bhk_outliers(df7)
df8.shape


# In[43]:


plot_scatter_chart(df8,"Hebbal")


# In[44]:


matplotlib.rcParams["figure.figsize"]=(20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("price_per_sqft")
plt.ylabel("count")


# In[45]:


df8.bath.unique()


# In[46]:


df8[df8.bath>10]


# In[47]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("bathroom")
plt.ylabel("count")


# In[48]:


df8[df8.bath>df8.bhk+2]


# In[49]:


df9=df8[df8.bath<df8.bhk+2]
df9.shape


# In[50]:


df10=df9.drop(["size","price_per_sqft"],axis="columns")
df10.head(3)


# # tut4

# In[51]:


dummies=pd.get_dummies(df10.location)
dummies.head()


# In[52]:


df11=pd.concat([df10,dummies.drop(['other'],axis="columns")],axis="columns")
df11.head()


# In[53]:


df12=df11.drop(["location"],axis="columns")
df12.head()


# In[54]:


df12.shape


# In[55]:


X=df12.drop(["price"],axis="columns")
y=df12["price"]
y.head()


# In[56]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)


# In[57]:


from sklearn.linear_model import LinearRegression
lr_clf=LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[58]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
cross_val_score(LinearRegression(),X,y,cv=cv)


# In[59]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
def find_best_model_using_gridsearchcv(X,y):
    algos={
        'linear_regression':{
            'model':LinearRegression(),
            'params':{
                'normalize':[True,False]
            }
        },
        'lasso':{
            'model':Lasso(),
            'params':{
                'alpha':[1,2],
                'selection':['random','cyclic']
            }
        },
        'decision_tree':{
            'model':DecisionTreeRegressor(),
            'params':{
                'criterion':['mse','friedman_mse'],
                'splitter':['best','random']
            }
        }
    }
    scores=[]
    cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algo_name,config in algos.items():
        gs=GridSearchCV(config['model'],config['params'],cv=cv,return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model':algo_name,
            'best_score':gs.best_score_,
            'best_params':gs.best_params_
        })
        
    return pd.DataFrame(scores,columns=["model","best_score","best_params"])
    
find_best_model_using_gridsearchcv(X,y)


# In[60]:


X.columns


# In[61]:


np.where(X.columns=='2nd Phase Judicial Layout')[0][0]


# In[62]:


def predict_price(location,sqft,bath,bhk):
    loc_index=np.where(X.columns==location)[0][0]
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if loc_index>=0:
        x[loc_index]=1
    return lr_clf.predict([x])


# In[63]:


predict_price('1st Phase JP Nagar',1000,2,2)[0]


# In[64]:


predict_price('1st Phase JP Nagar',1000,2,3)[0]


# In[65]:


predict_price('Indira Nagar',1000,2,2)[0]


# In[66]:


predict_price('Indira Nagar',1000,3,3)[0]


# In[67]:


import pickle
with open("banglore_home_price_model.pickle","wb") as f:
    pickle.dump(lr_clf,f)


# In[77]:


import json
columns={
    'data_columns' : [col.lower() for col in X.columns]
}
# columns
with open("banglore_home_price_model.json","w") as f:
    f.write(json.dumps(columns))


# In[ ]:




