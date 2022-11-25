import pandas as pd
import numpy as np
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
       # elif distance=='cosine':
        #    try:
         #       return np.dot(data,centroid)/(norm(data,axis=1)*norm(centroid))
          #  except:
           #     return (data*centroid).sum(axis=1)/(norm(data)*norm(centroid,axis=1))
        elif distance=='hamming':
            try:
                return abs(data-centroid).sum(axis=1)/data.shape[1]
            except:
                return abs(data-centroid).sum(axis=1)/data.shape[0]
        else:
            raise Exception("Los unet naziv distance")
    
    def init_centroids(self,data):
        centroids=data.sample(1).reset_index(drop=True)
        
      #centroids=pd.DataFrame(data.iloc[418]).T
        for cent in range(self.k-1):
            #data.apply(lambda x: self.calc_distance(self.distance,x,centroids))
            index_max_dist=np.argmax(data.apply(lambda x: self.calc_distance(self.distance,x,centroids).min(),axis=1))
            centroids=centroids.append(data.iloc[index_max_dist],ignore_index=True)
        return centroids
    
            #pomaaaa=data.apply(lambda x: ((x-centroids)**2).sum(axis=1).min(),axis=1).max()
            #a=data.apply(lambda x: np.dot(x,centroids)/(norm(x,axis=1)*norm(centroids)).min().max(),axis=1)
            #a.max()
            #pomdata.sort_values(by=[326],ascending=False)
            #pomdata.columns
            #pomdata.loc[380]
    def learn(self,data,iterations, weights):
        self.data_mean=data.mean()
        self.data_std=data.std()
        data=(data-self.data_mean)/self.data_std
        n,m=data.shape
        clusters=np.zeros((n,1))
        
        # inicijalizacija centroida
        #centroids=data.sample(self.k).reset_index(drop=True)
        #centroids=self.init_centroids(data)
        #inicijalizacija matrice distanci
        d=np.zeros((n,self.k))
        
        #old_SSE=np.float('inf')
        
        # Prolaz kroz iteracije 
        the_best_quality=np.float('inf')
        
        
        
        for try_iter in range(self.n_iteration):
            old_SSE=np.float('inf')
            #centroids=data.sample(self.k,random_state=1).reset_index(drop=True)
            centroids=self.init_centroids(data)
            for interation in range(iterations):
                # racunanje odstojanja
                quality=np.zeros(self.k)
                
                for centroid in range(self.k):
               #     #d[:,centroid]=((data-centroids.iloc[centroid])**2).sum(axis=1)
                    d[:,centroid]=self.calc_distance(self.distance,data,centroids.iloc[centroid]) # *weights
                
               # for i in range(n):
                #    slucaj=data.iloc[i]
                 #   dist=self.calc_distance(self.distance,slucaj,centroids)
                  #  clusters[i]=np.argmin(dist)
                
                
                # dodela klastera instancama
                clusters=np.argmin(d,axis=1)
            
                # racunanje srednjnih vrednosti za svaki atribut posebno
                for centroid in range(self.k):
                    centroids.iloc[centroid]=data[clusters==centroid].mean(axis=0)
                    quality[centroid]=(data[clusters==centroid].var()).sum()*len(data[clusters==centroid])
                    
                if quality.sum()==old_SSE: break #staviti the_best_total_quality
                old_SSE=quality.sum()
                print('old_sse           ',old_SSE)
                if old_SSE < the_best_quality:
                #    print('USao      ',the_best_quality)
                    the_best_quality=old_SSE
                    the_best_model=centroids
                    the_best_single_quality=quality
                
                    print("     ",old_SSE,the_best_quality)
                
                 #   print('USao      ',the_best_quality)
            #print("ITERARACIJA: ", try_iter, old_SSE.round(2),the_best_quality.round(2))
            #print("MODEL: ",centroids)
            print()
            print()
            print()
        #print("The best ",the_best_quality )
        #print("The best model ",the_best_model)
        return the_best_model,the_best_single_quality
    #%%
alg=Kmeans(distance='hamming')
print(alg.distance)
a,b= alg.learn(data, 50, [0.5,0.7,0.4,0.9,1.2,1.4,1.6,0.9,0.87,1,1,1.4,1.2,1.5])
b.sum()

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

