import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#from numpy.linalg import norm

data=pd.read_csv('data/boston.csv')
data.dtypes

#normalizacija

#%%
class Kmeans:
    
    def __init__ (self,k=3,distance='eucledian',n_iteration=7):
        self.k=k
        self.distance=distance
        self.calc_distance=self.type_of_distance
        self.n_iteration=n_iteration
        self.data_mean : float
        self.data_std : float
        
    def type_of_distance(self,distance, data, centroid):
        if distance=='eucledian':
            return ((data-centroid)**2).sum(axis=1) # ovde weights
        elif distance=='manhattan':
            return (abs(data-centroid)).sum(axis=1)
        elif distance=='chebychev':
            return (abs(data-centroid)).max(axis=1)
        elif distance=='hamming':
            try:
                return abs(data-centroid).sum(axis=1)/data.shape[1]
            except:
                return abs(data-centroid).sum(axis=1)/data.shape[0]
        else:
            raise Exception("Los unet naziv distance")
    
    def init_centroids(self,data):
        centroids=data.sample(1).reset_index(drop=True)        
        for cent in range(self.k-1):         
            index_max_dist=np.argmax(data.apply(lambda x: self.calc_distance(self.distance,x,centroids).min(),axis=1))
            centroids=centroids.append(data.iloc[index_max_dist],ignore_index=True )
        return centroids
    
    def learn(self,data,iterations, weights):
        self.data_mean=data.mean()
        self.data_std=data.std()
        data=(data-self.data_mean)/self.data_std
        n,m=data.shape
        clusters=np.zeros((n,1))
        
        d=np.zeros((n,self.k))
        
        the_best_quality=np.float('inf')
        
        
        
        for try_iter in range(self.n_iteration):
            old_SSE=np.float('inf')
            #centroids=data.sample(self.k,random_state=1).reset_index(drop=True)
            centroids=self.init_centroids(data)
            for interation in range(iterations):
                # racunanje odstojanja
                quality=np.zeros(self.k)
                
                for centroid in range(self.k):        
                    d[:,centroid]=self.calc_distance(self.distance,data,centroids.iloc[centroid]) 
                
                # dodela klastera instancama
                clusters=np.argmin(d,axis=1)
                
               # for i in range(n):
                #    slucaj=data.iloc[i]
                 #   dist=self.calc_distance(self.distance,slucaj,centroids)
                  #  clusters[i]=np.argmin(dist)
                
                # racunanje srednjnih vrednosti za svaki atribut posebno
                for centroid in range(self.k):
                    centroids.iloc[centroid]=data[clusters==centroid].mean(axis=0)
                    quality[centroid]=(data[clusters==centroid].var()).sum()*len(data[clusters==centroid])
                    
                if quality.sum()==old_SSE: break #staviti the_best_total_quality
                else: old_SSE=quality.sum()
            
                if old_SSE < the_best_quality:
                    the_best_quality=old_SSE
                    the_best_model=centroids
                    the_best_single_quality=quality
                    self.model=the_best_model
                    pom_c=clusters
                                
            print("ITERARACIJA: ", try_iter, old_SSE.round(2),the_best_quality.round(2))
            
        print("The best ",the_best_quality )
        print("The best model ",the_best_model)
        print(pom_c)
        return the_best_model,the_best_single_quality
    
    def predisct(self,data):
        data=(data-self.data_mean)/self.data_std
        n,m=data.shape
        d=np.zeros((n,self.k))
        for centroid in range(self.k):
            d[:,centroid]=self.calc_distance(self.distance,data,self.model.iloc[centroid])
            
            
        clusters=np.argmin(d,axis=1)
        data['Cluster']=clusters
        return data
            
    def silhouette_score(self,data,k):
        siluet=np.zeros(len(data))
        for i in range(len(data)):
            subset=data[data['Cluster']==data.iloc[i,-1]]
            a=((data.iloc[i,:-1]-subset.iloc[:,:-1])**2).sum(axis=1).sum()/(len(subset)-1)
            b= np.min([((data.iloc[i,:-1]-data[data['Cluster']==j].iloc[:,:-1])**2).sum(axis=1).sum()/sum(data['Cluster']==j) for j in range(k) if data.iloc[i,-1]!=j])
            siluet[i]=(b-a)/max(b,a)
        return siluet.mean(),siluet
    
    def the_best_k(self,data,k_range):
        scores={}
        for k in k_range:
            model=Kmeans(k)
            model.learn(data, 3, np.ones(data.shape[1]))
            pom_data=model.predisct(data)
            #scores.append(self.silhouette_score(pom_data,k))            
            scores[k],_=self.silhouette_score(pom_data,k)    
        return scores   
        
    def get_information(self,data):
        udaljenosti={}
        _,siluets=self.silhouette_score(data, self.k)
        data['Siluet']=siluets
        siluet_scors=data.groupby('Cluster')['Siluet'].agg(['mean'])
        for k in range(self.k-1):
            udaljenosti[k]=((self.model.iloc[k]-self.model.loc[k+1:])**2).sum(axis=1)
        return udaljenosti,siluet_scors
    

#%%
alg=Kmeans(distance='eucledian')

the_best_k=alg.the_best_k(data, [2,3])

#Vrati najbolje k pu siluet skoru
#the_best_k=max(the_best_k,key=the_best_k.get)

print(alg.distance)

a,b= alg.learn(data, 50, [0.4,0.5,0.2,0.9,1.3,1.5,1.5,0.8,1.4,1.2,1,1,1.2,1.5])

pom_data=alg.predisct(data)

u,s=alg.get_information(pom_data)

# informacije:
#Kvalitet svakog klastera
b
#Udaljenost centroida
u
#Siluet skorovi
s



siluet=np.zeros(len(pom_data))
for i in range(len(pom_data)):
    subset=pom_data[pom_data['Cluster']==pom_data.iloc[i,-1]]
    a=((pom_data.iloc[i,:-1]-subset.iloc[:,:-1])**2).sum(axis=1).sum()/(len(subset)-1)
    b= np.min([((pom_data.iloc[i,:-1]-pom_data[pom_data['Cluster']==j].iloc[:,:-1])**2).sum(axis=1).sum()/(sum(pom_data['Cluster']==j)-1)  for j in range(3) if pom_data.iloc[i,-1]!=j])
    siluet[i]=(b-a)/max(b,a)
pom_data['Siluet']=siluet    

pom_data.groupby('Cluster')['Siluet'].agg('mean')
pom_data['Siluet'].mean()



cl=0
pom_data['Cluster']
for cl in range(3):
    subset=pom_data[pom_data['Cluster']==cl]
    subset1=pom_data[pom_data['Cluster']!=cl]
    pom_subset=subset.iloc[:,:-1].apply(lambda x: ((x-subset.iloc[:,:-1])**2).sum().sum()/len(subset),axis=1)
    subset.iloc[:,:-1].apply(lambda x: ((x-subset1.iloc[:,:-1])**2).sum().sum()/len(subset),axis=1)
x=subset.iloc[0,:-1]
len(pom_data.loc[pom_data['Cluster']==0])


siluet=np.zeros(len(pom_data))
i=0

subsets={}
for cl in range(3):
    subsets[cl]=pom_data[pom_data['Cluster']==cl]

subsets[0]
j=0
i=0
#siluet skor
for i in range(len(pom_data)):
    subset=pom_data[pom_data['Cluster']==pom_data.iloc[i,-1]]
    a=((pom_data.iloc[i,:-1]-subset.iloc[:,:-1])**2).sum(axis=1).sum()/(len(subset)-1)
    b= np.min([((pom_data.iloc[i,:-1]-pom_data[pom_data['Cluster']==j].iloc[:,:-1])**2).sum(axis=1).sum()/(sum(pom_data['Cluster']==j)-1)  for j in range(3) if pom_data.iloc[i,-1]!=j])
    siluet[i]=(b-a)/max(b,a)
pom_data['Siluet']=siluet    

pom_data.groupby('Cluster')['Siluet'].agg('mean')

max(siluet[siluet>0])

pom_data[pom_data['Cluster']==j].iloc[:,:-1]

sum(pom_data['Cluster']==j)
pom_data[pom_data['Cluster']==j]

((pom_data.iloc[i,:-1]-pom_data[pom_data['Cluster']==j].iloc[:,:-1])**2).sum().sum()

#pom_data.loc[pom_data['Cluster']==0]['Cluster']

b.sum()

alg.model.iloc[0]

alg.learn(data, 50, np.ones(14))


np.array([0.5,0.7,0.4,0.9,1.2,1.4,1.6,0.9,0.87,1,1,1.4,1.2,1.5]).sum()

pom=np.array([[1,2],[2,3]])
pom_cen=np.array([[0.5,0.5],[0.3,0.2]])
(pom-pom_cen).sum(axis=1)
k=0

#udaljenost izmedju klastera
udaljenosti={}
for k in range(2):
    udaljenosti[k]=((centroids.iloc[k]-centroids.loc[k+1:])**2).sum(axis=1)


# zato sto kad se pomnozei nesto ako nije dobro raposedljeno po tome, bice velika greska
4211


i=0
scores=np.zeros(shape=3)
for i in range(k):
    temp_data=pom_data[pom_data['Cluster']==i]
    
    a=alg.calc_distance('eucledian',temp_data, temp_data.mean(axis=0))/(temp_data.shape[0]-1)
    b=np.min([alg.calc_distance('eucledian',temp_data, pom_data[pom_data['Cluster']==j].mean(axis=0))/(temp_data.shape[0]) for j in range(3) if not i==j])    
    scores[int(i)]=(b-a)/max(b,a,axis=1)
    if(information):
        self.visualize(scores[int(i)],i)

