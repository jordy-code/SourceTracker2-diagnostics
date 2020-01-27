#!/usr/bin/ python
#-*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:50:32 2020

@author: MehCracken
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pandas_ml import ConfusionMatrix

# Read in the data
df=pd.read_csv('RF bacteria table.csv')
# Split the data set into test and training set
df['is_train']=np.random.uniform(0, 1, len(df)) <= .75


''' The domain_df is used to determine important features for each environment
For each environment you want to test, remove that environment name from the
regex list below, all remaining environments in the list will be considered
'other'.
For instance, if you want to know important features for gut, remove 'gut'
from the regex list and change the domain variable to the domain and the environent
to be tested'''
domain='viruses_gut'
domain_df=df.replace(regex=['Sand','CoastalMarine','Soil','Freshwater'], value='Other')
print(domain_df)


'''This is the function to run the random forrest'''
def random_forest(df):
    train, test = df[df['is_train']==True], df[df['is_train']==False]
    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:',len(test))
    features=df.columns[1:]

    y=pd.factorize(train['Env'])[0]
    clf = RandomForestClassifier(n_estimators=500, random_state=0)
    clf.fit(train[features], y)

    target_names=df['Env'].unique()
    preds=target_names[clf.predict(test[features])]
    
    # confusion matrix array
    cmatrix=pd.crosstab(test['Env'], preds, rownames=['Actual Env'], colnames=['Predicted Env'])

    #important features
    importance=list(zip(train[features], clf.feature_importances_))
    importance_sorted=sorted(importance, key=lambda x: x[1], reverse=True)
    
    ffeature=importance_sorted[:20]
    feature_df=pd.DataFrame(ffeature, columns=['feature','importance'])
    feature_df.to_csv(domain+'_features.csv')
    
    #Important features graphs
    ffeature.reverse()
    plt.plot([val[1] for val in ffeature], range(len(ffeature)), 'o')
    plt.hlines(range(len(ffeature)), [0], [val[1] for val in ffeature], 
               linestyles='dotted', lw=2)
    plt.yticks(range(len(ffeature)), [val[0] for val in ffeature])
    plt.tight_layout()
    plt.savefig(domain+'_graph.png', dpi=300)
    
    
    ''' This 'cm' vaiable is what we used to create the confusion matrices. It uses
    pandas_ml which is not compatible with pandas version 0.25. Pandas needs
    to be downgraded to 0.24.2 in order for this to work. The 'cmatrix' variable above
    will also produce a matrix for the confusion matrix but pandas_ml was used originally 
    to create the graph. If you comment these 3 lines out, the random forest function will work
    and produce the important features.'''
    cm = ConfusionMatrix(test["Env"].tolist(), preds)
    cm.print_stats()
    cm.plot()

'''Depending on whether you are running the random forest for all environments to produce 
confusion matrices or if you want to know important features for individual environments, input
needs to be changed. The 'df' vriable uses all the environments to either classify
an individual sample or produce confusion matrices. The 'domain_df' variable labels 
chosen environments as 'other' to determine important features in classifying individual envrionmets 
from all other environments. This 'domain_df' variable is briefly described above.'''

random_forest(df)


