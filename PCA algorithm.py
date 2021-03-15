import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
#library for pca
from sklearn import preprocessing
import matplotlib.pyplot as plt

genes=['gene'+str(i) for i in range(1,101)]

wt=['wt'+str(i) for i in range(1,6)]
ko=['ko'+str(i) for i in range(1,6)]

data=pd.DataFrame(columns=[*wt,*ko],index =genes)
#[wt1,wt2...,wt5,ko1,ko2,...ko5] bcz we used star
#if we didn't use star then it will create 2D array like [[wt1..wt5],[ko1...ko5]]
#so use Stars

for gene in data.index:
    data.loc[gene,'wt1':'wt5']=np.random.poisson(lam=rd.randrange(10,100),size=5)
    data.loc[gene,'ko1':'ko5']=np.random.poisson(lam=rd.randrange(10,100),size=5)
print(data.head())
print(data.shape)

scaled_data=preprocessing.scale(data.T)

pca=PCA()
pca.fit(scaled_data)
pca_data=pca.transform(scaled_data)

#percentage of variation that each PCS accounts for
per_var=np.round(pca.explained_variance_ratio_*100,decimals=1)
#labels for scree plot
labels=['PC' +str(x) for x in range(1,len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1),height= per_var ,tick_label=labels)
plt.ylabel("percentage of Explained Variance")
plt.xlabel('Principle component')
plt.title('Scree Plot')
plt.show()

#draw pca plot
pca_df=pd.DataFrame(pca_data,index=[*wt,*ko],columns=labels)
pca_df.head()

plt.scatter(pca_df.PC1,pca_df.PC2)
plt.title('pca graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0])) #in PCA indexing starts with 0 that's why
plt.ylabel("PC2 - {0}%".format(per_var[1]))
for sample in pca_df.index:
    plt.annotate(sample,(pca_df.PC1.loc[sample]),(pca_df.PC2.loc[sample]))
    #this loop to add sample name to graph
plt.show()    

loading_scores= pd.Series(pca.components_[0],index=genes)
sorted_v=loading_scores.abs().sort_values(ascending=False)
#abs = absolute (magnitude)
top10=sorted_v[0:10].index.values
print(loading_scores[top10])
