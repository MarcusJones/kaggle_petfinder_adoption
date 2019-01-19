# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] {"_uuid": "3b5c9dc5eedea5af75e07cada3328992f7e7dc84"}
# ## General information
#
# This kernel is dedicated to EDA of PetFinder.my Adoption Prediction challenge as well as feature engineering and modelling.
#
# ![](https://i.imgur.com/rvSWCYO.png)
# (a screenshot of the PetFinder.my site)
#
# In this dataset we have lots of information: tabular data, texts and even images! This gives a lot of possibilties for feature engineering and modelling. The only limiting factor is the fact that the competition is kernel-only. On the other hand this will ensure everyone has the same computational resources.
#
# In this kernel I want to pay attention to several things:
# * comparing distribution of features in train and test data;
# * exploring features and their interactions;
# * trying various types of feature engineering;
# * trying various models without neural nets (for now);
#
# It is important to remember that this competition has stage 2, so our models will run against unseen data.
#
# *Work still in progress*

# %% {"_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5", "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19", "_kg_hide-input": true}
#libraries
import numpy as np 
import pandas as pd 
import os
import json
import seaborn as sns 
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('ggplot')
import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from PIL import Image
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import gc
from catboost import CatBoostClassifier
from tqdm import tqdm_notebook
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import random
import warnings
warnings.filterwarnings("ignore")
from functools import partial
pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 100)
import os
import scipy as sp
from math import sqrt
from collections import Counter
from sklearn.metrics import confusion_matrix as sk_cmatrix

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
from sklearn.ensemble import RandomForestClassifier
import langdetect
import eli5
from IPython.display import display 

from sklearn.metrics import cohen_kappa_score
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

# %% {"_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0", "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a", "_kg_hide-input": true}
breeds = pd.read_csv('../input/breed_labels.csv')
colors = pd.read_csv('../input/color_labels.csv')
states = pd.read_csv('../input/state_labels.csv')

train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')
sub = pd.read_csv('../input/test/sample_submission.csv')

train['dataset_type'] = 'train'
test['dataset_type'] = 'test'
all_data = pd.concat([train, test])

# %% [markdown] {"_uuid": "64c7b97d92588795fe9968d0d4d0ada723fe3a9d"}
# ## Data overview
#
# Let's have a quick look at the data first!

# %% {"_uuid": "f636d2b0986b026e2f9e05b1e088ea3e8f0f2e3f"}
print(os.listdir("../input"))

# %% {"_uuid": "aeae0a045c4e8bc12ec95b954666cf57442fdfac"}
train.drop('Description', axis=1).head()

# %% {"_uuid": "9faa1ff39a080f5ab524bc9fecbccb2777628139"}
train.info()

# %% [markdown] {"_uuid": "0a9533822d60c67f1efd4d536ffd781e84d8678e"}
# * We have almost 15 thousands dogs and cats in the dataset;
# * Main dataset contains all important information about pets: age, breed, color, some characteristics and other things;
# * Desctiptions were analyzed using Google's Natural Language API providing sentiments and entities. I suppose we could do a similar thing ourselves;
# * There are photos of some pets;
# * Some meta-information was extracted from images and we can use it;
# * There are separate files with labels for breeds, colors and states;
#
# Let's start with the main dataset.
#
# I have also created a full dataset by combining train and test data. This is done purely for more convenient visualization. Column "dataset_type" shows which dataset the data belongs to.

# %% [markdown] {"_uuid": "8542f37472712951dc614bd5d4fce2403b2efff2"}
# ## Main data exploration

# %% [markdown] {"_uuid": "d2352291c16f881f9dbd851e2dbed3cfe1e0b15c"}
# ### Target: Adoption speed
#
# * 0 - Pet was adopted on the same day as it was listed.
# * 1 - Pet was adopted between 1 and 7 days (1st week) after being listed.
# * 2 - Pet was adopted between 8 and 30 days (1st month) after being listed.
# * 3 - Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed.
# * 4 - No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days). 

# %% {"_uuid": "14c64e761f2f5df96b90bc82d20d57c67004e38d", "_kg_hide-input": true}
train['AdoptionSpeed'].value_counts().sort_index().plot('barh', color='teal');
plt.title('Adoption speed classes counts');

# %% [markdown] {"_uuid": "605f45c4f67fe32115799cf269c6a232e2eef615"}
# A small note on how annotating works:
# * When I use seaborn countplot, I assign the figure to a variable - this allows to change its attributes and go deeper into its parameters;
# * Figure has `Axes` - bars - which contain information about color, transparency and other parameters;
# * And `patches` in `Axes` contain this information;
# * So we can take information from 'patches`, for example width and height of each bar, and plot correct text in correct places
#
# https://matplotlib.org/users/artists.html

# %% {"_kg_hide-output": true, "_uuid": "a02eef2ccad681aa0103cbd22a8e8032af74489e"}
plt.figure(figsize=(14, 6));
g = sns.countplot(x='AdoptionSpeed', data=all_data.loc[all_data['dataset_type'] == 'train']);
plt.title('Adoption speed classes rates');
ax=g.axes

# %% {"_uuid": "d6da75eddb7b7938b1accf9812ebae34f6cf2e61"}
#Axes
ax

# %% {"_uuid": "f62c09b5bb3f0b75fb98b919efdcbc8a5a84aaae"}
# patches
ax.patches

# %% {"_uuid": "e25d2efe8c0fd2d3aa7ad00e80fccca41cf13fd9"}
# example of info in patches
ax.patches[0].get_x()

# %% {"_kg_hide-input": true, "_uuid": "0529ee4fa802f9c0575b8f6e88533bc4a94746af"}
plt.figure(figsize=(14, 6));
g = sns.countplot(x='AdoptionSpeed', data=all_data.loc[all_data['dataset_type'] == 'train'])
plt.title('Adoption speed classes rates');
ax=g.axes
for p in ax.patches:
     ax.annotate(f"{p.get_height() / train.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
         textcoords='offset points')  

# %% [markdown] {"_uuid": "16b66de28b50c41c8fcc2c2ea5d77e0e722f6658"}
# We can see that some pets were adopted immediately, but these are rare cases: maybe someone wanted to adopt any pet, or the pet was lucky to be seen by person, who wanted a similar pet.
# A lot of pets aren't adopted at all, which is quite sad :( I hope our models and analysis will help them to find their home!
#
# It is nice that a lot of pets are adopted within a first week of being listed!
#
# One more interesting thing is that the classes have a linear relationship - the higher the number, the worse situation is. So it could be possible to build not only multiclass classification, but also regression.

# %% [markdown] {"_uuid": "ea232a5cf54c3220f266863e29b9f8517fda6b3a"}
# ### Type
# 1 - Dog, 2 - Cat

# %% {"_uuid": "af73348f3044d637a5f11bc4645a31d725b6670e", "_kg_hide-input": true}
all_data['Type'] = all_data['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')
plt.figure(figsize=(10, 6));
sns.countplot(x='dataset_type', data=all_data, hue='Type');
plt.title('Number of cats and dogs in train and test data');

# %% [markdown] {"_uuid": "0765c6567da19a6bd83789421675e29338bbb02b"}
# We can see that the rate of dogs in train dataset is higher that in test set. But I don't think the difference is seriuos.

# %% [markdown] {"_uuid": "6ec4fe56f1e6126b833549a954c27a8a74a7e572"}
# #### Comparison of rates
#
# From here on I'll compare not only counts of pets in different categories, but also compate adoption speed rates with base ones.
#
# This is how it works:
# * As we saw earlier the base rate of pets with Adoption speed 0 is 410 / 14993 = 0.027;
# * Now look at the next graph: there are 6861 cats in train dataset and 240 of them have Adoption Speed 0. So the rate is 240 / 6861 = 0.035;
# * 0.035 / 0027 = 1.2792 so cats have 27, 92% more chances to have Adoption Speed 0;

# %% {"_kg_hide-input": true, "_uuid": "5ac22d86fb4266d9ba39645a910bcbf0ff6a8db7"}
main_count = train['AdoptionSpeed'].value_counts(normalize=True).sort_index()
def prepare_plot_dict(df, col, main_count):
    """
    Preparing dictionary with data for plotting.
    
    I want to show how much higher/lower are the rates of Adoption speed for the current column comparing to base values (as described higher),
    At first I calculate base rates, then for each category in the column I calculate rates of Adoption speed and find difference with the base rates.
    
    """
    main_count = dict(main_count)
    plot_dict = {}
    for i in df[col].unique():
        val_count = dict(df.loc[df[col] == i, 'AdoptionSpeed'].value_counts().sort_index())

        for k, v in main_count.items():
            if k in val_count:
                plot_dict[val_count[k]] = ((val_count[k] / sum(val_count.values())) / main_count[k]) * 100 - 100
            else:
                plot_dict[0] = 0

    return plot_dict

def make_count_plot(df, x, hue='AdoptionSpeed', title='', main_count=main_count):
    """
    Plotting countplot with correct annotations.
    """
    g = sns.countplot(x=x, data=df, hue=hue);
    plt.title(f'AdoptionSpeed {title}');
    ax = g.axes

    plot_dict = prepare_plot_dict(df, x, main_count)

    for p in ax.patches:
        h = p.get_height() if str(p.get_height()) != 'nan' else 0
        text = f"{plot_dict[h]:.0f}%" if plot_dict[h] < 0 else f"+{plot_dict[h]:.0f}%"
        ax.annotate(text, (p.get_x() + p.get_width() / 2., h),
             ha='center', va='center', fontsize=11, color='green' if plot_dict[h] > 0 else 'red', rotation=0, xytext=(0, 10),
             textcoords='offset points')  

# %% {"_uuid": "e45a3d16efb767b4de299f5d28527e76a493e118"}
plt.figure(figsize=(18, 8));
make_count_plot(df=all_data.loc[all_data['dataset_type'] == 'train'], x='Type', title='by pet Type')

# %% [markdown] {"_uuid": "479f2dd53a07fc87e05579b54cda20e4f9a9844c"}
# We can see that cats are more likely to be adopted early than dogs and overall the percentage of not adopted cats is lower. Does this mean people prefer cats? Or maybe this dataset is small and could contain bias.
# On the other hand more dogs are adopted after several months.

# %% [markdown] {"_uuid": "c33e9eb0b3a712e5d9c2d46591055377ebcd1e34"}
# ### Name
# I don't really think that names are important in adoption, but let's see.
#
# At first let's look at most common names.

# %% {"_kg_hide-input": true, "_uuid": "aba94b4f396cac1c5151c5984318ed26e1b3dc64"}
fig, ax = plt.subplots(figsize = (16, 12))
plt.subplot(1, 2, 1)
text_cat = ' '.join(all_data.loc[all_data['Type'] == 'Cat', 'Name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white',
                      width=1200, height=1000).generate(text_cat)
plt.imshow(wordcloud)
plt.title('Top cat names')
plt.axis("off")

plt.subplot(1, 2, 2)
text_dog = ' '.join(all_data.loc[all_data['Type'] == 'Dog', 'Name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white',
                      width=1200, height=1000).generate(text_dog)
plt.imshow(wordcloud)
plt.title('Top dog names')
plt.axis("off")

plt.show()

# %% [markdown] {"_uuid": "82d458fa1b622dba68d3e244dd0b8d97191d8c66"}
# Cute names! :) I like some of them!
#
# It is worth noticing some things:
# * Often we see normal pet names like "Mimi", "Angel" and so on;
# * Quite often people write simply who is there for adoption: "Kitten", "Puppies";
# * Vety often the color of pet is written, sometimes gender;
# * And it seems that sometimes names can be strange or there is some info written instead of the name;
#
# One more thing to notice is that some pets don't have names. Let's see whether this is important

# %% {"_uuid": "4c3b1a154876b224f9de77af9103bfc0686ffb87"}
print('Most popular pet names and AdoptionSpeed')
for n in train['Name'].value_counts().index[:5]:
    print(n)
    print(train.loc[train['Name'] == n, 'AdoptionSpeed'].value_counts().sort_index())
    print('')

# %% [markdown] {"_uuid": "e50b97ac02bb49c8445dd52916d0c2c318aa8638"}
# #### No name

# %% {"_kg_hide-input": true, "_uuid": "7673019ef3d7ed0f03db97a770c88343b1429d7b"}
train['Name'] = train['Name'].fillna('Unnamed')
test['Name'] = test['Name'].fillna('Unnamed')
all_data['Name'] = all_data['Name'].fillna('Unnamed')

train['No_name'] = 0
train.loc[train['Name'] == 'Unnamed', 'No_name'] = 1
test['No_name'] = 0
test.loc[test['Name'] == 'Unnamed', 'No_name'] = 1
all_data['No_name'] = 0
all_data.loc[all_data['Name'] == 'Unnamed', 'No_name'] = 1

print(f"Rate of unnamed pets in train data: {train['No_name'].sum() * 100 / train['No_name'].shape[0]:.4f}%.")
print(f"Rate of unnamed pets in test data: {test['No_name'].sum() * 100 / test['No_name'].shape[0]:.4f}%.")

# %% {"_uuid": "c0d12229204c3173d61d9fd50725be7d347e60bb"}
pd.crosstab(train['No_name'], train['AdoptionSpeed'], normalize='index')

# %% [markdown] {"_uuid": "9ccbfd235c0f1f1c49a28a59380a12ce4bba80bb"}
# Less than 10% of pets don't have names, but they have a higher possibility of not being adopted.

# %% {"_uuid": "1d91e8d3b3233dd60f7d4f2edd284bc24df2d63d", "_kg_hide-input": true}
plt.figure(figsize=(18, 8));
make_count_plot(df=all_data.loc[all_data['dataset_type'] == 'train'], x='No_name', title='and having a name')

# %% [markdown] {"_uuid": "441b5440d73f116986cffe7700e2954ce5116b51"}
# #### "Bad" names
#
# I have noticed that shorter names tend to be meaningless. Here is an example of some names with 3 characters.

# %% {"_uuid": "ec8345055a0a8c0b8f1cdbecfeee07d4fdf3b4ba"}
all_data[all_data['Name'].apply(lambda x: len(str(x))) == 3]['Name'].value_counts().tail()

# %% [markdown] {"_uuid": "63cb97e8bb01902129db7bddb9e9207a7f8e9a7b"}
# And here are names with 1 or 2 characters...

# %% {"_uuid": "c62f444899bc2a956d582de306b1c93c447b0293"}
all_data[all_data['Name'].apply(lambda x: len(str(x))) < 3]['Name'].unique()

# %% [markdown] {"_uuid": "74fc6b4f2832b522601d1cd78fbeb99924eec4fb"}
# I think that we could create a new feature, showing that name is meaningless - pets with these names could have less success in adoption.

# %% [markdown] {"_uuid": "89eb4bc124932ffc778bb954b1045b2878a56cd9"}
# ### Age

# %% {"_uuid": "b2bf602e384fefb192a91f2ded29e902df5b1483", "_kg_hide-input": true}
fig, ax = plt.subplots(figsize = (16, 6))
plt.subplot(1, 2, 1)
plt.title('Distribution of pets age');
train['Age'].plot('hist', label='train');
test['Age'].plot('hist', label='test');
plt.legend();

plt.subplot(1, 2, 2)
plt.title('Distribution of pets age (log)');
np.log1p(train['Age']).plot('hist', label='train');
np.log1p(test['Age']).plot('hist', label='test');
plt.legend();

# %% {"_uuid": "f931e84ad40d1acdf65b469a0eac3c7dbc279b79"}
train['Age'].value_counts().head(10)

# %% [markdown] {"_uuid": "f34bcab946dfbed17f83358055e2a2f141129b8e"}
# We can see that most pets are young - maybe after the birth. Also there a lot of pets with an age equal to multiples of 12 - I think than owners didn't bother with the exact age.

# %% {"_uuid": "0882a131ea8dc53772aaa77a32ae39eaebad4f7b", "_kg_hide-input": true}
plt.figure(figsize=(10, 6));
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and age');

# %% {"_uuid": "8aeb4bcc7295e6e5e47b12b6da135ea2e81b295f", "_kg_hide-input": true}
data = []
for a in range(5):
    df = train.loc[train['AdoptionSpeed'] == a]

    data.append(go.Scatter(
        x = df['Age'].value_counts().sort_index().index,
        y = df['Age'].value_counts().sort_index().values,
        name = str(a)
    ))
    
layout = go.Layout(dict(title = "AdoptionSpeed trends by Age",
                  xaxis = dict(title = 'Age (months)'),
                  yaxis = dict(title = 'Counts'),
                  )
                  )
py.iplot(dict(data=data, layout=layout), filename='basic-line')

# %% [markdown] {"_uuid": "b5dd32edf666aded299493fe341d2bf387fd15ef"}
# * We can see that young pets are adopted quite fast and most of them are adopted;
# * Most pets are less than 4 months old with a huge spike at 2 months;
# * It seems that a lot of people don't input exact age and write age in years (or multiples of 12);
# * It could make sense to create some binary variables based on age;

# %% [markdown] {"_uuid": "0a98ebc0ed3511b8661d01f385af1500830dec97"}
# ### Breeds
# There is a main breed of the pet and secondary if relevant.
#
# At first let's see whether having secondary breed influences adoption speed.

# %% {"_uuid": "d881bcc7cb65c714ca61e55b70ee1c9fa2bb4b54", "_kg_hide-input": true}
train['Pure_breed'] = 0
train.loc[train['Breed2'] == 0, 'Pure_breed'] = 1
test['Pure_breed'] = 0
test.loc[test['Breed2'] == 0, 'Pure_breed'] = 1
all_data['Pure_breed'] = 0
all_data.loc[all_data['Breed2'] == 0, 'Pure_breed'] = 1

print(f"Rate of pure breed pets in train data: {train['Pure_breed'].sum() * 100 / train['Pure_breed'].shape[0]:.4f}%.")
print(f"Rate of pure breed pets in test data: {test['Pure_breed'].sum() * 100 / test['Pure_breed'].shape[0]:.4f}%.")

# %% {"_kg_hide-input": true, "_uuid": "3a4d0fbca9b371b56e940ff4576bc9a5fa55c089"}
def plot_four_graphs(col='', main_title='', dataset_title=''):
    """
    Plotting four graphs:
    - adoption speed by variable;
    - counts of categories in the variable in train and test;
    - adoption speed by variable for dogs;
    - adoption speed by variable for cats;    
    """
    plt.figure(figsize=(20, 12));
    plt.subplot(2, 2, 1)
    make_count_plot(df=train, x=col, title=f'and {main_title}')

    plt.subplot(2, 2, 2)
    sns.countplot(x='dataset_type', data=all_data, hue=col);
    plt.title(dataset_title);

    plt.subplot(2, 2, 3)
    make_count_plot(df=train.loc[train['Type'] == 1], x=col, title=f'and {main_title} for dogs')

    plt.subplot(2, 2, 4)
    make_count_plot(df=train.loc[train['Type'] == 2], x=col, title=f'and {main_title} for cats')
    
plot_four_graphs(col='Pure_breed', main_title='having pure breed', dataset_title='Number of pets by pure/not-pure breed in train and test data')

# %% [markdown] {"_uuid": "958ca3062f38c5ef33ec6b01bb39d0a4c91e810d"}
# It seems that non-pure breed pets tend to be adopted more and faster, especially cats.
#
# Let's look at the breeds themselves

# %% {"_uuid": "a1e930d8f1bb50ec1735adbd05c1f22f3d13ab51"}
breeds_dict = {k: v for k, v in zip(breeds['BreedID'], breeds['BreedName'])}

# %% {"_kg_hide-input": true, "_uuid": "637edff87ab2b77e500782e779b3732a109009c5"}
train['Breed1_name'] = train['Breed1'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
train['Breed2_name'] = train['Breed2'].apply(lambda x: '_'.join(breeds_dict[x]) if x in breeds_dict else '-')

test['Breed1_name'] = test['Breed1'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
test['Breed2_name'] = test['Breed2'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else '-')

all_data['Breed1_name'] = all_data['Breed1'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
all_data['Breed2_name'] = all_data['Breed2'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else '-')

# %% {"_uuid": "dd739c34bf9311ca357159ec978a7201a6b436dd", "_kg_hide-input": true}
fig, ax = plt.subplots(figsize = (20, 18))
plt.subplot(2, 2, 1)
text_cat1 = ' '.join(all_data.loc[all_data['Type'] == 'Cat', 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_cat1)
plt.imshow(wordcloud)
plt.title('Top cat breed1')
plt.axis("off")

plt.subplot(2, 2, 2)
text_dog1 = ' '.join(all_data.loc[all_data['Type'] == 'Dog', 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_dog1)
plt.imshow(wordcloud)
plt.title('Top dog breed1')
plt.axis("off")

plt.subplot(2, 2, 3)
text_cat2 = ' '.join(all_data.loc[all_data['Type'] == 'Cat', 'Breed2_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_cat2)
plt.imshow(wordcloud)
plt.title('Top cat breed1')
plt.axis("off")

plt.subplot(2, 2, 4)
text_dog2 = ' '.join(all_data.loc[all_data['Type'] == 'Dog', 'Breed2_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,
                      width=1200, height=1000).generate(text_dog2)
plt.imshow(wordcloud)
plt.title('Top dog breed2')
plt.axis("off")
plt.show()

# %% [markdown] {"_uuid": "e7c9ec2abb3d60a8ec682063052f62acd762f463"}
# It seems that not all values of these features are really breeds. Sometimes people simply write that the dogs has a mixed breed, cats often are described as domestic with certain hair length.
#
# Now let's have a look at the combinations of breed names.

# %% {"_uuid": "b21482f03c2bf3bb2eba684147dda7d777013762"}
(all_data['Breed1_name'] + '__' + all_data['Breed2_name']).value_counts().head(15)

# %% [markdown] {"_uuid": "aff14ca84a79dddda64b202e628e8bb7dbb32207"}
# It seems that most dogs aren't pure breeds, but mixed breeds! My first assumption was wrong.
#
# Sometimes people write "mixed breed" in the first fiels, sometimes in both, and sometimes main breed is in the first field and is marked as mixed breed in the second field.
#
# I think we can create new features based on this information. And later we can verify the hair length of pets.

# %% [markdown] {"_uuid": "56bdade2a8b9c3ea1d282f23a5801df80add345e"}
# ### Gender
#  1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets

# %% {"_uuid": "181f0075c215c25c0b6346e4468dd47b1ccb7d72", "_kg_hide-input": true}
plt.figure(figsize=(18, 6));
plt.subplot(1, 2, 1)
make_count_plot(df=train, x='Gender', title='and gender')

plt.subplot(1, 2, 2)
sns.countplot(x='dataset_type', data=all_data, hue='Gender');
plt.title('Number of pets by gender in train and test data');

# %% {"_uuid": "f70e78d4cef95fe9058b7de6ae30093d39559bbd", "_kg_hide-input": true}
sns.factorplot('Type', col='Gender', data=all_data, kind='count', hue='dataset_type');
plt.subplots_adjust(top=0.8)
plt.suptitle('Count of cats and dogs in train and test set by gender');

# %% [markdown] {"_uuid": "a82b61384813d4208c8cbf9e68e79b58074ec9ee"}
# It seems that male pets are adopted faster than female. Having no information about the gender really decreases chances.

# %% [markdown] {"_uuid": "53fa658b6224b3dfcd7836fe866ac7866f0dad9f"}
# ### Colors

# %% {"_kg_hide-input": true, "_uuid": "5183e7f289e686e17c05d6f91f2d7e4529640684"}
colors_dict = {k: v for k, v in zip(colors['ColorID'], colors['ColorName'])}
train['Color1_name'] = train['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
train['Color2_name'] = train['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
train['Color3_name'] = train['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

test['Color1_name'] = test['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
test['Color2_name'] = test['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
test['Color3_name'] = test['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

all_data['Color1_name'] = all_data['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
all_data['Color2_name'] = all_data['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
all_data['Color3_name'] = all_data['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

# %% {"_kg_hide-input": true, "_uuid": "ed0156512e6f8b8d22d02a74bc46849761b6c4ca"}
def make_factor_plot(df, x, col, title, main_count=main_count, hue=None, ann=True, col_wrap=4):
    """
    Plotting countplot.
    Making annotations is a bit more complicated, because we need to iterate over axes.
    """
    if hue:
        g = sns.factorplot(col, col=x, data=df, kind='count', col_wrap=col_wrap, hue=hue);
    else:
        g = sns.factorplot(col, col=x, data=df, kind='count', col_wrap=col_wrap);
    plt.subplots_adjust(top=0.9);
    plt.suptitle(title);
    ax = g.axes
    plot_dict = prepare_plot_dict(df, x, main_count)
    if ann:
        for a in ax:
            for p in a.patches:
                text = f"{plot_dict[p.get_height()]:.0f}%" if plot_dict[p.get_height()] < 0 else f"+{plot_dict[p.get_height()]:.0f}%"
                a.annotate(text, (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='green' if plot_dict[p.get_height()] > 0 else 'red', rotation=0, xytext=(0, 10),
                     textcoords='offset points')  

# %% {"_uuid": "fe07002920afa569d6a1339be28601aa09297ee4", "_kg_hide-input": true}
sns.factorplot('dataset_type', col='Type', data=all_data, kind='count', hue='Color1_name', palette=['Black', 'Brown', '#FFFDD0', 'Gray', 'Gold', 'White', 'Yellow']);
plt.subplots_adjust(top=0.8)
plt.suptitle('Counts of pets in datasets by main color');

# %% [markdown] {"_uuid": "55903a9cd5c54cfab6b8dfe7452b4e0456b26cb8"}
# We can see that most common colors are black and brown. Interesting to notice that there are almost no gray or yellow dogs :)
#
# Now let's see whether colors influence adoption speed

# %% {"_kg_hide-input": true, "_uuid": "0b9879b5c9832d002c4d7472cb3989a12a2a1a40"}
make_factor_plot(df=train, x='Color1_name', col='AdoptionSpeed', title='Counts of pets by main color and Adoption Speed')

# %% {"_kg_hide-input": true, "_uuid": "fd0a21adc084dad9d53865f18bedc844a7d21539"}
train['full_color'] = (train['Color1_name'] + '__' + train['Color2_name'] + '__' + train['Color3_name']).str.replace('__', '')
test['full_color'] = (test['Color1_name'] + '__' + test['Color2_name'] + '__' + test['Color3_name']).str.replace('__', '')
all_data['full_color'] = (all_data['Color1_name'] + '__' + all_data['Color2_name'] + '__' + all_data['Color3_name']).str.replace('__', '')

make_factor_plot(df=train.loc[train['full_color'].isin(list(train['full_color'].value_counts().index)[:12])], x='full_color', col='AdoptionSpeed', title='Counts of pets by color and Adoption Speed')

# %% [markdown] {"_uuid": "10e2273417de407e4fb104cd10a813ba0f5429a9"}
# We can see that there are some differences based on color, but the number of pets in most colors isn't very high, so this could be due to randomness.

# %% {"_uuid": "7fba74b81cd002cf2b867b586c7d2f69c6cdbc45"}
gender_dict = {1: 'Male', 2: 'Female', 3: 'Mixed'}
for i in all_data['Type'].unique():
    for j in all_data['Gender'].unique():
        df = all_data.loc[(all_data['Type'] == i) & (all_data['Gender'] == j)]
        top_colors = list(df['full_color'].value_counts().index)[:5]
        j = gender_dict[j]
        print(f"Most popular colors of {j} {i}s: {' '.join(top_colors)}")

# %% [markdown] {"_uuid": "783219e4add952b649a21f60129561594dd881e4"}
# ### MatiritySize
# Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)

# %% {"_kg_hide-input": true, "_uuid": "4177da0415552a52ecc1a26a9f6811d11bbdda4a"}
plot_four_graphs(col='MaturitySize', main_title='MaturitySize', dataset_title='Number of pets by MaturitySize in train and test data')

# %% {"_uuid": "3b11d2a8afbb3bfd89643e38b3e29b06f90c1edf"}
make_factor_plot(df=all_data, x='MaturitySize', col='Type', title='Count of cats and dogs in train and test set by MaturitySize', hue='dataset_type', ann=False)

# %% {"_uuid": "31159a85ecb67be76bd9955b1c8fa39288720d5a", "_kg_hide-input": true}
images = [i.split('-')[0] for i in os.listdir('../input/train_images/')]
size_dict = {1: 'Small', 2: 'Medium', 3: 'Large', 4: 'Extra Large'}
for t in all_data['Type'].unique():
    for m in all_data['MaturitySize'].unique():
        df = all_data.loc[(all_data['Type'] == t) & (all_data['MaturitySize'] == m)]
        top_breeds = list(df['Breed1_name'].value_counts().index)[:5]
        m = size_dict[m]
        print(f"Most common Breeds of {m} {t}s:")
        
        fig = plt.figure(figsize=(25, 4))
        
        for i, breed in enumerate(top_breeds):
            # excluding pets without pictures
            b_df = df.loc[(df['Breed1_name'] == breed) & (df['PetID'].isin(images)), 'PetID']
            if len(b_df) > 1:
                pet_id = b_df.values[1]
            else:
                pet_id = b_df.values[0]
            ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])

            im = Image.open("../input/train_images/" + pet_id + '-1.jpg')
            plt.imshow(im)
            ax.set_title(f'Breed: {breed}')
        plt.show();

# %% [markdown] {"_uuid": "7d9e4759c640f24c6135387acf8e8742a6d97499"}
# Quite interesting:
# * We can see that maturity size isn't very important. Medium sized pets are most common and they have slightly more chances to be not adopted;
# * There are almost no Extra Large pets. I hope it means that their owners like them and there is no need for them to be adopted :)
# * I wanted to gave a look at different pets, so I showed examples of pictures of most common breeds for each maturity size of cats and dogs;
# * I think not all data is entirely correct: sometimes short haired cats have breed with "medium hair", not sure that all breeds are entirely correct. Some photoes have bad quality;

# %% [markdown] {"_uuid": "fa5fada835211e4ee296af3f4d62a38cf9bb398a"}
# ### FurLength
#
#  (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)

# %% {"_kg_hide-input": true, "_uuid": "0c1d309ef612f2b821a444fee1e58c17feb95cc5"}
plot_four_graphs(col='FurLength', main_title='FurLength', dataset_title='Number of pets by FurLength in train and test data')

# %% [markdown] {"_uuid": "054f00a5304987efd024fd88da21a5d9d12ff4ec"}
# * We can see that most of the pets have short fur and long fur is the least common;
# * Pets with long hair tend to have a higher chance of being adopted. Though it could be because of randomness due to low count;
#
# As I said earlier, some breed have hair length in the text, let's check these values!

# %% {"_kg_hide-input": true, "_uuid": "9b8382cc28ae0069adea06df4eca3bf11fb91c07"}
fig, ax = plt.subplots(figsize = (20, 18))
plt.subplot(2, 2, 1)
text_cat1 = ' '.join(all_data.loc[(all_data['FurLength'] == 1) & (all_data['Type'] == 'Cat'), 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text_cat1)
plt.imshow(wordcloud)
plt.title('Top cat breed1 with short fur')
plt.axis("off")

plt.subplot(2, 2, 2)
text_dog1 = ' '.join(all_data.loc[(all_data['FurLength'] == 1) & (all_data['Type'] == 'Dog'), 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text_dog1)
plt.imshow(wordcloud)
plt.title('Top dog breed1 with short fur')
plt.axis("off")

plt.subplot(2, 2, 3)
text_cat2 = ' '.join(all_data.loc[(all_data['FurLength'] == 2) & (all_data['Type'] == 'Cat'), 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text_cat2)
plt.imshow(wordcloud)
plt.title('Top cat breed1 with medium fur')
plt.axis("off")

plt.subplot(2, 2, 4)
text_dog2 = ' '.join(all_data.loc[(all_data['FurLength'] == 2) & (all_data['Type'] == 'Dog'), 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1200, height=1000).generate(text_dog2)
plt.imshow(wordcloud)
plt.title('Top dog breed2 with medium fur')
plt.axis("off")
plt.show()

# %% {"_uuid": "d7217972bb283763ab900f27d24b64a78ac37e20"}
c = 0
strange_pets = []
for i, row in all_data[all_data['Breed1_name'].str.contains('air')].iterrows():
    if 'Short' in row['Breed1_name'] and row['FurLength'] == 1:
        pass
    elif 'Medium' in row['Breed1_name'] and row['FurLength'] == 2:
        pass
    elif 'Long' in row['Breed1_name'] and row['FurLength'] == 3:
        pass
    else:
        c += 1
        strange_pets.append((row['PetID'], row['Breed1_name'], row['FurLength']))
        
print(f"There are {c} pets whose breed and fur length don't match")

# %% [markdown] {"_uuid": "e6564e8ec0301db1b3e43d228eaae6b578a64c9c"}
# It seems that almost one thousand pets have mismatch in breeds and fur lengths. Let's see!

# %% {"_kg_hide-output": false, "_uuid": "7ac3b519ffb4abeb0d2648ad1639700cdbe26254", "_kg_hide-input": true}
strange_pets = [p for p in strange_pets if p[0] in images]
fig = plt.figure(figsize=(25, 12))
fur_dict = {1: 'Short', 2: 'Medium', 3: 'long'}
for i, s in enumerate(random.sample(strange_pets, 12)):
    ax = fig.add_subplot(3, 4, i+1, xticks=[], yticks=[])

    im = Image.open("../input/train_images/" + s[0] + '-1.jpg')
    plt.imshow(im)
    ax.set_title(f'Breed: {s[1]} \n Fur length: {fur_dict[s[2]]}')
plt.show();

# %% [markdown] {"_uuid": "348e99f3ba2817ff981e1a21ee7d491089ec5fd0"}
# Everybody lies!
#
# Sometimes breed is more correct, sometimes fur length... I suppose we could create a feature showing whether breed and fur length match.

# %% [markdown] {"_uuid": "906ae8c4cc259edca8130d320bd179b409d56469"}
# ### Health
#
# There are four features showing health of the pets:
#
# * Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
# * Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
# * Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
# * Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
#
# I think that these features are very important - most people would prefer a healthy pet. While sterilization isn't the main concern, having healty and dewormed pet should have a great importance. Let's see whether I'm right!

# %% {"_kg_hide-input": true, "_uuid": "6b367f78ca433579f61d06411e5b361c41342cf5"}
plt.figure(figsize=(20, 12));
plt.subplot(2, 2, 1)
make_count_plot(df=train, x='Vaccinated', title='Vaccinated')
plt.xticks([0, 1, 2], ['Yes', 'No', 'Not sure']);
plt.title('AdoptionSpeed and Vaccinated');

plt.subplot(2, 2, 2)
make_count_plot(df=train, x='Dewormed', title='Dewormed')
plt.xticks([0, 1, 2], ['Yes', 'No', 'Not sure']);
plt.title('AdoptionSpeed and Dewormed');

plt.subplot(2, 2, 3)
make_count_plot(df=train, x='Sterilized', title='Sterilized')
plt.xticks([0, 1, 2], ['Yes', 'No', 'Not sure']);
plt.title('AdoptionSpeed and Sterilized');

plt.subplot(2, 2, 4)
make_count_plot(df=train, x='Health', title='Health')
plt.xticks([0, 1, 2], ['Healthy', 'Minor Injury', 'Serious Injury']);
plt.title('AdoptionSpeed and Health');

plt.suptitle('Adoption Speed and health conditions');

# %% [markdown] {"_uuid": "91d081672c5db05488a1ef84871834528ab1d6ea"}
# * Almost all pets are healthy! Pets with minor injuries are rare and sadly they aren't adopted well. Number of pets with serious injuries is negligible.
# * It is interesting that people prefer non-vaccinated pets. Maybe they want to bring pets to vets themselves...
# * People also prefer non-sterilized pets! Maybe they want puppies/kittens :)
# * Quite important is the fact that when there is no information about health condition, the probability of not being adopted is much higher;
#
# Let's have a look at most popular health conditions.

# %% {"_uuid": "299e53c93240a700e706e8ea7b4ae4b103874490", "_kg_hide-input": true}
train['health'] = train['Vaccinated'].astype(str) + '_' + train['Dewormed'].astype(str) + '_' + train['Sterilized'].astype(str) + '_' + train['Health'].astype(str)
test['health'] = test['Vaccinated'].astype(str) + '_' + test['Dewormed'].astype(str) + '_' + test['Sterilized'].astype(str) + '_' + test['Health'].astype(str)


make_factor_plot(df=train.loc[train['health'].isin(list(train.health.value_counts().index[:5]))], x='health', col='AdoptionSpeed', title='Counts of pets by main health conditions and Adoption Speed')

# %% [markdown] {"_uuid": "a78d5e1ac3afc054c096c46114eb8fd08e8d410f"}
# * Healthy, dewormed and non-sterilized pets tend to be adopted faster!
# * Completely healthy pets are... more likely to be not adopted! I suppose that means that a lot of people pay attention to other characteristics;
# * And healthy pets with no information (not sure value) also tend to be adopted less frequently. Maybe people prefer having information, even if it is negative;

# %% {"_kg_hide-input": true, "_uuid": "c10a37f98966145d1780897159bdaf8349d66c86"}
plt.figure(figsize=(20, 16))
plt.subplot(3, 2, 1)
sns.violinplot(x="AdoptionSpeed", y="Age", data=train);
plt.title('Age distribution by Age');
plt.subplot(3, 2, 3)
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Vaccinated", data=train);
plt.title('Age distribution by Age and Vaccinated');
plt.subplot(3, 2, 4)
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Dewormed", data=train);
plt.title('Age distribution by Age and Dewormed');
plt.subplot(3, 2, 5)
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Sterilized", data=train);
plt.title('Age distribution by Age and Sterilized');
plt.subplot(3, 2, 6)
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Health", data=train);
plt.title('Age distribution by Age and Health');

# %% [markdown] {"_uuid": "c07691860afd4035a7c33b3e45ed1c0c21d59011"}
# ### Quantity
# Sometimes there are several pets in one advertisement.

# %% {"_uuid": "1cfb828f5854393030b87974f7e6f2d24bdd24d5"}
train.loc[train['Quantity'] > 11][['Name', 'Description', 'Quantity', 'AdoptionSpeed']].head(10)

# %% {"_uuid": "b0ea6c320c15bd2c0dcbd17e1a4d9f08c3cf7654"}
train['Quantity'].value_counts().head(10)

# %% [markdown] {"_uuid": "01f5574b61acf106b19c2855099483e5f711228f"}
# Sometimes there is a huge amount of pets in some advertisements! But at the same time sometimes text and the quantity don't match. For example:
#
#     Pancho and Tita are 2 adorable, playful kittens. They can be shy at first but once they get to know you they are the sweetest pets anyone could ask for. Available for adoption now. They are very, very close so we are looking for someone who can take them both.
#     
#  Obvously there are only two kittens, but the quantity is 12 for some reason.
#  
#  One thing worth noticing that sometimes all these pet are adopted which is great!
#  
#  For the sake of plotting I'll create a new variable, where 6 pets in one advertizement will the the max amount.

# %% {"_uuid": "e9b21e72515a61989ed3d4411ad3d95b62e21719", "_kg_hide-input": true}
train['Quantity_short'] = train['Quantity'].apply(lambda x: x if x <= 5 else 6)
test['Quantity_short'] = test['Quantity'].apply(lambda x: x if x <= 5 else 6)
all_data['Quantity_short'] = all_data['Quantity'].apply(lambda x: x if x <= 5 else 6)
plot_four_graphs(col='Quantity_short', main_title='Quantity_short', dataset_title='Number of pets by Quantity_short in train and test data')

# %% [markdown] {"_uuid": "b992a49cb96e8d93e1bce3308bbbb4a59b9bcfe1"}
# It seems that quantity has little to do with adoption speed. This is good, it means that abandoned cats/dogs with kittens/puppies have chances of being adopted! Though it seems that single cats have somewhat higher chances that single dogs.

# %% [markdown] {"_uuid": "9a3b8c45c502a6b777214583b268176a77cbd8b8"}
# ### Fee
# One of interesting features is adoption fee. Some pets can be gotten for free, adopting some required paying a certain amount.

# %% {"_kg_hide-input": true, "_uuid": "e7591ff01f25232af38c73d0d96e53a99bfeb34f"}
train['Free'] = train['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
test['Free'] = test['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
all_data['Free'] = all_data['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
plot_four_graphs(col='Free', main_title='Free', dataset_title='Number of pets by Free in train and test data')

# %% [markdown] {"_uuid": "88b34d852bff73eb5a72ab3c74acb4ac7cec08ae"}
# Most pets are free and it seems that asking for a fee slightly desreased the chance of adoption. Also free cats are adopted faster than free dogs

# %% {"_uuid": "fc2836121b091a830569858250d013f1c0600676"}
all_data.sort_values('Fee', ascending=False)[['Name', 'Description', 'Fee', 'AdoptionSpeed', 'dataset_type']].head(10)

# %% {"_kg_hide-input": true, "_uuid": "264d23cd4e7108893523d028c86333287024a347"}
plt.figure(figsize=(16, 6));
plt.subplot(1, 2, 1)
plt.hist(train.loc[train['Fee'] < 400, 'Fee']);
plt.title('Distribution of fees lower than 400');

plt.subplot(1, 2, 2)
sns.violinplot(x="AdoptionSpeed", y="Fee", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and Fee');

# %% [markdown] {"_uuid": "dca7d0cc1eb204ae1c62b105c45b4bbf09722f04"}
# * It is interesting that pets with high fee tend to be adopted quite fast! Maybe people prefer to pay for "better" pets: healthy, trained and so on;
# * Most pets are given for free and fees are usually lower than 100 $;
# * Fees for dogs tend to be higher, though these are rare cases anyway.

# %% {"_uuid": "42f2447ed4197ca1e950b348846b2383cef7be0f"}
plt.figure(figsize=(16, 10));
sns.scatterplot(x="Fee", y="Quantity", hue="Type",data=all_data);
plt.title('Quantity of pets and Fee');

# %% [markdown] {"_uuid": "97b35254458779abbad1f104ddca741a80322017"}
# It seems that fees and pet quantity have inversely proportional relationship. The less pets, the higher is the fee. I suppose these single pets are better trained and prepared than most others.

# %% [markdown] {"_uuid": "e69b241bbbe4e1d5ff929c698819deff4dfa1e0e"}
# ### State

# %% {"_uuid": "71c41bfeb3acc076ba41307617f89788f98faddb"}
states_dict = {k: v for k, v in zip(states['StateID'], states['StateName'])}
train['State_name'] = train['State'].apply(lambda x: '_'.join(states_dict[x].split()) if x in states_dict else 'Unknown')
test['State_name'] = test['State'].apply(lambda x: '_'.join(states_dict[x].split()) if x in states_dict else 'Unknown')
all_data['State_name'] = all_data['State'].apply(lambda x: '_'.join(states_dict[x].split()) if x in states_dict else 'Unknown')

# %% {"_uuid": "b2962de32eacf5c3f1c292cfd60ce62797f34586"}
all_data['State_name'].value_counts(normalize=True).head()

# %% [markdown] {"_uuid": "40fff8f5d21c513fcd46bea5b5c9d7e45ae1a8f4"}
# Sadly I don't know anything about Malaysia’s states, so I can only say that top three states account for ~90% of ads. Let's have a look at them.

# %% {"_kg_hide-input": true, "_uuid": "1ebe6886e3b92c62c71ff816125af2600d695b1f"}
make_factor_plot(df=train.loc[train['State_name'].isin(list(train.State_name.value_counts().index[:3]))], x='State_name', col='AdoptionSpeed', title='Counts of pets by states and Adoption Speed')

# %% [markdown] {"_uuid": "6c26bbd4e4365406d2581d2ab37d86dd097e2890"}
# Intetestingly top-2 and top-3 states have lower rates of adoption.

# %% [markdown] {"_uuid": "057d283ff616d2b563c75cdf3f291362b632cfa8"}
# ### Rescuer
# We have unique hashes for resquers.

# %% {"_uuid": "483574c727e7f65d68e695617a821c833057e978"}
all_data['RescuerID'].value_counts().head()

# %% [markdown] {"_uuid": "a2cfc9049c68cdb8c67be8b949d1b607fb5ca166"}
# Top-5 resquers managed a lot of pets!
# I wonder whether these are individual people or organizations. Let's have a look at them.

# %% {"_uuid": "141b33aa83e333b3aa4f908c9248854ae439994c"}
make_factor_plot(df=train.loc[train['RescuerID'].isin(list(train.RescuerID.value_counts().index[:5]))], x='RescuerID', col='AdoptionSpeed', title='Counts of pets by rescuers and Adoption Speed', col_wrap=5)

# %% [markdown] {"_uuid": "04062628d1a72209c30400d35f363e812eaedbcd"}
# Wow! The resquer with the highest amount of resqued pets has the best adoption rate! On the other hand the third one has the worst rate :(

# %% [markdown] {"_uuid": "bd666d9ba51eec84858a71847fa2a0e3bde99838"}
# ### VideoAmt

# %% {"_uuid": "89e1bbe4f8e738f2ade6fed5ed0c3c82d1b040d5"}
train['VideoAmt'].value_counts()

# %% [markdown] {"_uuid": "176bc84b64a5d61b9eb98b0fbdc299a8471381f2"}
# Hm. In most cases there are no videos at all. Sometimes there is one video, more than one video is quite rare. We don't have videos and considering a huge disbalance in values I'm not sure this variable will be useful.

# %% [markdown] {"_uuid": "e8836c571f74b3989d9a1ae5bfc15ed550d57990"}
# ### PhotoAmt

# %% {"_uuid": "f8badaeb2e79f0050d72f90c94397c7a8f1842c4"}
print(F'Maximum amount of photos in {train["PhotoAmt"].max()}')
train['PhotoAmt'].value_counts().head()

# %% {"_uuid": "435b3cca2715fff5d00d4b43a1e613a33d5d29e6"}
make_factor_plot(df=train.loc[train['PhotoAmt'].isin(list(train.PhotoAmt.value_counts().index[:5]))], x='PhotoAmt', col='AdoptionSpeed', title='Counts of pets by PhotoAmt and Adoption Speed', col_wrap=5)

# %% {"_uuid": "15828acbcc3497661de2cfaa21900ceca2d3ff34"}
plt.figure(figsize=(16, 6));
plt.subplot(1, 2, 1)
plt.hist(train['PhotoAmt']);
plt.title('Distribution of PhotoAmt');

plt.subplot(1, 2, 2)
sns.violinplot(x="AdoptionSpeed", y="PhotoAmt", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and PhotoAmt');

# %% [markdown] {"_uuid": "4c16b3a846d088db84e435fa200091ecfbf102ac"}
# Pets can have up to 30 photos! That's a lot! But I'm not convinced that amount of photoes has any real influence.

# %% [markdown] {"_uuid": "7d4f53076823b79dbcd682966791df6b97e34be2"}
# ### Description
#
# Description contains a lot of important information, let' analyze it!

# %% {"_uuid": "e3e6878f01fa17bf368219e89df0d9199b99bd03"}
fig, ax = plt.subplots(figsize = (12, 8))
text_cat = ' '.join(all_data['Description'].fillna('').values)
wordcloud = WordCloud(max_font_size=None, background_color='white',
                      width=1200, height=1000).generate(text_cat)
plt.imshow(wordcloud)
plt.title('Top words in description');
plt.axis("off");

# %% [markdown] {"_uuid": "995a87dd48d98da789414b19074470d4bb8151a5"}
# There are too many similar general words like "cat". We need to go deeper.
#
# Let's use ELI5 library for prediction explanation. I'll fit a basic vectorizer on desctriptions and build a simple Random Forest model. Then we will look at words which caused certain labels to be predicted.

# %% {"_kg_hide-input": true, "_uuid": "e2cc8749825baef6989ca7bc3113b61dcdcbe2d1"}
tokenizer = TweetTokenizer()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)

vectorizer.fit(all_data['Description'].fillna('').values)
X_train = vectorizer.transform(train['Description'].fillna(''))

rf = RandomForestClassifier(n_estimators=20)
rf.fit(X_train, train['AdoptionSpeed'])

# %% {"_uuid": "53be7de4dba82fb3ac7f14498a713cf1bfc6b2e3"}
for i in range(5):
    print(f'Example of Adoption speed {i}')
    text = train.loc[train['AdoptionSpeed'] == i, 'Description'].values[0]
    print(text)
    display(eli5.show_prediction(rf, doc=text, vec=vectorizer, top=10))

# %% [markdown] {"_uuid": "44718d6d4468a88ce36af396fc1a5b590c5806fa"}
# Some words/phrases seem to be useful, but it seems that different adoption speed classes could have similar important words...

# %% {"_kg_hide-input": true, "_uuid": "dedb813860df7549409694e0de219488a58c6de4"}
train['Description'] = train['Description'].fillna('')
test['Description'] = test['Description'].fillna('')
all_data['Description'] = all_data['Description'].fillna('')

train['desc_length'] = train['Description'].apply(lambda x: len(x))
train['desc_words'] = train['Description'].apply(lambda x: len(x.split()))

test['desc_length'] = test['Description'].apply(lambda x: len(x))
test['desc_words'] = test['Description'].apply(lambda x: len(x.split()))

all_data['desc_length'] = all_data['Description'].apply(lambda x: len(x))
all_data['desc_words'] = all_data['Description'].apply(lambda x: len(x.split()))

# %% {"_kg_hide-input": true, "_uuid": "3bb458e11f583ac254f919543f63c714a81b9c45"}
plt.figure(figsize=(16, 6));
plt.subplot(1, 2, 1)
sns.violinplot(x="AdoptionSpeed", y="desc_length", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and description length');

plt.subplot(1, 2, 2)
sns.violinplot(x="AdoptionSpeed", y="desc_words", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and count of words in description');

# %% [markdown] {"_uuid": "6cfaa461767bd287786cc51fe157f840e6b0bcd7"}
# Interestingly pets with short text in ads are adopted quickly. Or maybe longer descriptions mean more problems in the pets, therefore adoption speed is lower?

# %% [markdown] {"_uuid": "43a1670ae93a7a704437fb8468b597bc68fe11a4"}
# ### Sentiment
# We have run each pet profile's description through Google's Natural Language API, providing analysis on sentiment and key entities. You may optionally utilize this supplementary information for your pet description analysis. There are some descriptions that the API could not analyze. As such, there are fewer sentiment files than there are rows in the dataset. 

# %% {"_kg_hide-input": true, "_uuid": "7da27e65952a4b19d9e63de56facb17c6c74add7"}
sentiment_dict = {}
for filename in os.listdir('../input/train_sentiment/'):
    with open('../input/train_sentiment/' + filename, 'r') as f:
        sentiment = json.load(f)
    pet_id = filename.split('.')[0]
    sentiment_dict[pet_id] = {}
    sentiment_dict[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']
    sentiment_dict[pet_id]['score'] = sentiment['documentSentiment']['score']
    sentiment_dict[pet_id]['language'] = sentiment['language']

for filename in os.listdir('../input/test_sentiment/'):
    with open('../input/test_sentiment/' + filename, 'r') as f:
        sentiment = json.load(f)
    pet_id = filename.split('.')[0]
    sentiment_dict[pet_id] = {}
    sentiment_dict[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']
    sentiment_dict[pet_id]['score'] = sentiment['documentSentiment']['score']
    sentiment_dict[pet_id]['language'] = sentiment['language']

# %% {"_kg_hide-input": true, "_uuid": "409a1b79f29012e37b6d4e9bbe15fba1715ce6ee"}
train['lang'] = train['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
train['magnitude'] = train['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
train['score'] = train['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

test['lang'] = test['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
test['magnitude'] = test['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
test['score'] = test['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

all_data['lang'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
all_data['magnitude'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
all_data['score'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

# %% {"_kg_hide-input": true, "_uuid": "657f352d865016a7d5fa9220d59fb1520ecca563"}
plot_four_graphs(col='lang', main_title='lang', dataset_title='Number of pets by lang in train and test data')

# %% [markdown] {"_uuid": "2a4a735a9403cc409e081b77591bcd02bd29f9b6"}
# Well, English is the most common language by far, so language feature will hardly help.

# %% {"_uuid": "7e79ac98ddad0776a1f3d21538abd31a020ff5ad"}
plt.figure(figsize=(16, 6));
plt.subplot(1, 2, 1)
sns.violinplot(x="AdoptionSpeed", y="score", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and score');

plt.subplot(1, 2, 2)
sns.violinplot(x="AdoptionSpeed", y="magnitude", hue="Type", data=train);
plt.title('AdoptionSpeed by Type and magnitude of sentiment');

# %% [markdown] {"_uuid": "b554e95c200d22a00a6f2c2d7c6c5ba52dc76e65"}
# It seems that the lower is the magnitude of score, the faster pets are adopted.

# %% [markdown] {"_uuid": "c90159a95d5952c4b077bac75d450b89010a4737"}
# ### Basic model
#
# There are much more interesting things in the dataset and I'm going to explore them, but for now let's build a simple model as a baseline.

# %% {"_uuid": "20b467c6f19c9eac301e233acbc2d5a0f3c32f09"}
cols_to_use = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'No_name', 'Pure_breed', 'health', 'Free', 'desc_length', 'desc_words', 'score', 'magnitude']
train = train[[col for col in cols_to_use if col in train.columns]]
test = test[[col for col in cols_to_use if col in test.columns]]

# %% {"_uuid": "a4841f7e73af45347802752d59556ea3e9d4523f"}
cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'State', 'RescuerID',
       'No_name', 'Pure_breed', 'health', 'Free']

# %% {"_uuid": "d136fe07e55e9b008f9856898e3d4a26e0f941ba"}
more_cols = []
for col1 in cat_cols:
    for col2 in cat_cols:
        if col1 != col2 and col1 not in ['RescuerID', 'State'] and col2 not in ['RescuerID', 'State']:
            train[col1 + '_' + col2] = train[col1].astype(str) + '_' + train[col2].astype(str)
            test[col1 + '_' + col2] = test[col1].astype(str) + '_' + test[col2].astype(str)
            more_cols.append(col1 + '_' + col2)
            
cat_cols = cat_cols + more_cols

# %% {"_uuid": "d5d341bb001c3ae0009d1362773840fe85b9609c"}
# %%time
indexer = {}
for col in cat_cols:
    # print(col)
    _, indexer[col] = pd.factorize(train[col].astype(str))
    
for col in tqdm_notebook(cat_cols):
    # print(col)
    train[col] = indexer[col].get_indexer(train[col].astype(str))
    test[col] = indexer[col].get_indexer(test[col].astype(str))


# %% {"_uuid": "26d823fffcdd7830e9224a5c22de58b51502b9b8"}
y = train['AdoptionSpeed']
train = train.drop(['AdoptionSpeed'], axis=1)

# %% [markdown] {"_uuid": "ccb79707784bf821af808fcf820611649078cce9"}
# ## Naive multiclass LGB

# %% {"_uuid": "9ac32335b35a7aa014fd87a1e9d0ac932f372459"}
n_fold = 5
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=15)

# %% {"_uuid": "64e95e47a595b47083bd6dd3d99ffebb53fc6443", "_kg_hide-input": true}
def train_model(X=train, X_test=test, y=y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, averaging='usual', make_oof=False):
    result_dict = {}
    if make_oof:
        oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        gc.collect()
        print('Fold', fold_n + 1, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        
        if model_type == 'lgb':
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature = cat_cols)
            valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature = cat_cols)
            
            model = lgb.train(params,
                    train_data,
                    num_boost_round=2000,
                    valid_sets = [train_data, valid_data],
                    verbose_eval=100,
                    early_stopping_rounds = 200)

            del train_data, valid_data
            
            y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration).argmax(1)
            del X_valid
            gc.collect()
            y_pred = model.predict(X_test, num_iteration=model.best_iteration).argmax(1)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)
            
        if model_type == 'lcv':
            model = LogisticRegressionCV(scoring='neg_log_loss', cv=3, multi_class='multinomial')
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
            
        if model_type == 'cat':
            model = CatBoostClassifier(iterations=20000,  loss_function='MultiClass', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test).reshape(-1,)
        
        if make_oof:
            oof[valid_index] = y_pred_valid.reshape(-1,)
            
        scores.append(kappa(y_valid, y_pred_valid))
        print('Fold kappa:', kappa(y_valid, y_pred_valid))
        print('')
        
        if averaging == 'usual':
            prediction += y_pred
        elif averaging == 'rank':
            prediction += pd.Series(y_pred).rank().values
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importance()
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'lgb':
        
        if plot_feature_importance:
            feature_importance["importance"] /= n_fold
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
            
            result_dict['feature_importance'] = feature_importance
            
    result_dict['prediction'] = prediction
    if make_oof:
        result_dict['oof'] = oof
    
    return result_dict

# %% {"_uuid": "5af7d16419babb3af2c4b10f429aa65bc1bdd2b1"}
params = {'num_leaves': 128,
        #  'min_data_in_leaf': 60,
         'objective': 'multiclass',
         'max_depth': -1,
         'learning_rate': 0.05,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 3,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
        #  "lambda_l1": 0.1,
         # "lambda_l2": 0.1,
         "random_state": 42,          
         "verbosity": -1,
         "num_class": 5}

# %% {"_uuid": "85db5cb12bad162e2c3a0f08c640382a1228d73d"}
result_dict_lgb = train_model(X=train, X_test=test, y=y, params=params, model_type='lgb', plot_feature_importance=True, make_oof=True)

# %% {"_uuid": "73325a3247f9cafd3af95323de9d4933a82e4d0b"}
xgb_params = {'eta': 0.01, 'max_depth': 10, 'subsample': 0.9, 'colsample_bytree': 0.9, 
          'objective': 'multi:softmax', 'eval_metric': 'merror', 'silent': True, 'nthread': 4, 'num_class': 5}
result_dict_xgb = train_model(params=xgb_params, model_type='xgb', make_oof=True)

# %% {"_uuid": "e5fcff4424ad30b5dafd6fda1eaac811a69aef5b"}
prediction = (result_dict_lgb['prediction'] + result_dict_xgb['prediction']) / 2
submission = pd.DataFrame({'PetID': sub.PetID, 'AdoptionSpeed': [int(i) for i in prediction]})
submission.head()

# %% {"_uuid": "5f4dd1fbd5fa4e6ed3f4fa7ea140cb9197bf6f94"}
submission.to_csv('submission.csv', index=False)
