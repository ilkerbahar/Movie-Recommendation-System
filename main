import numpy as np
import labelx as labelx
import pandas as pd
import collections
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import tkinter as tk
from tkinter import simpledialog
import math as m

root = tk.Tk()
root.title(" MOVIE RECOMMENDATION SYSTEM BLM3120 TERM PROJECT")
frame = tk. Frame (root, bg="#3e646c", padx=19, pady=11)
frame.pack(padx=20,pady=12)
columns = ['user_id', 'item_id', 'rating', 'timestamp']
userRatings = pd.read_csv('u.data', sep='\t', names=columns)
userRatings.drop('timestamp', axis=1, inplace = True)
movieColumns = ['item_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies = pd.read_csv('u.item', sep='|', names=movieColumns, encoding='latin-1')
movieNames = movies[['item_id', 'movie title']]

df= pd.merge(userRatings,movieNames, on ='item_id')



trainData, testData = train_test_split(df, test_size=0.25)

ratingTrain = trainData.pivot_table(values='rating', index='user_id', columns='movie title',fill_value=0)
ratingTest = testData.pivot_table(values='rating', index='user_id', columns='movie title', fill_value=0)
watched = collections.defaultdict(dict)
for i in df.values.tolist():
    watched [i[0]][i[1]] = i[2]
userMatrix = csr_matrix(ratingTrain.values) 


knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
knn.fit(userMatrix)
k = 25





def userBasedPredict(userid):

    
    usr_idx= userid-1
    distances, indices = knn.kneighbors (ratingTrain.iloc[usr_idx, :]\
                        .values.reshape(1, -1), n_neighbors = k)
    user_watched = set(watched[userid])
    neighborsWatched = {}


    for i in range(0, len(distances. flatten())):
        neighborsWatched [ratingTrain.index[indices.flatten()[i]]] = watched[ratingTrain.index[indices.flatten()[i]]].copy()

        for key, val in neighborsWatched [ratingTrain.index[indices.flatten()[i]]].items():
            neighborsWatched[ratingTrain.index[indices.flatten()[i]]][key] = [1-distances. flatten()[i], val]
    
    
    unwatchedMovies = []

   

    for u in neighborsWatched:
        a = neighborsWatched[u].keys() - user_watched.intersection (neighborsWatched [u].keys())
        for f in a:
            unwatchedMovies.append(f)
        commonUnwatched = [item for item, count in collections.Counter(unwatchedMovies).items() if count > 1]
        commonUnwatchedRating = []
        for f in commonUnwatched:
            a=[]
            w=[]
            for u in neighborsWatched:
                if neighborsWatched[u].get(f) is not None:
                    a.append(neighborsWatched [u].get(f)[0]*neighborsWatched [u].get(f)[1])
                    w.append(neighborsWatched [u].get(f)[0])
            commonUnwatchedRating.append([np.sum(a)/np.sum(w),f])
    commonUnwatchedRating = sorted (commonUnwatchedRating, reverse=True)
    return commonUnwatchedRating



def userBased():
    
    INPUT = tk.simpledialog.askinteger(title = "User Based CF", prompt="Enter the user id : ")
    r = userBasedPredict(INPUT)
    message = "Suggestions for user #" + str(INPUT) + "\n"
    for f in r[:10]:
        message += "{0} - {1} - {2:.2f}".format(f[1], movieNames.loc[movieNames['item_id'] == f[1]]['movie title'].values[0], f[0]) + "\n"
        sum=0
        n=0.00001


  
    for f in r[:200]:
        if (ratingTest.at[INPUT-1, movieNames.loc[movieNames['item_id'] == f[1]]['movie title'].values[0]] != 0):
            sum += pow(f[0] - ratingTest.loc [INPUT-1, movieNames.loc[movieNames['item_id'] == f[1]]['movie title'].values[0]], 2)
            n+=1
        else:
            pass
    mse = sum/n
    rmse = m.sqrt(mse)
    message += "RMSE : " + str(rmse) + "\n" + "MSE : "+ str(mse) #MSE--> mean squared error  RMSE--> Root Mean Squared Error
    labelx.config(text = message)
    labelx.pack()

def itemBasedPredict(moviename, ratings):
    
    SVD = TruncatedSVD(n_components=12, random_state=5) 
    resultant_matrix = SVD.fit_transform(ratings.T)
    
    corr_mat = np.corrcoef (resultant_matrix)
    col_idx = ratings.columns.get_loc (moviename)
    corr_specific = corr_mat[col_idx]
    k = pd.DataFrame({ 'corr_specific':corr_specific, 'Movies': ratings.columns})\
    .sort_values('corr_specific', ascending=False)
    return k['Movies'].head(11)


def itemBased():
  
    INPUT = tk.simpledialog.askstring(title = "Item Based CF",prompt = "Enter the movie name: ")
    result = itemBasedPredict (INPUT,ratingTrain)
    message = "Top 10 suggestions for people who watched " + str(INPUT) + "\n"
    for i in range (1,11):
        message += result.iloc[i] + "\n"

    labelx.config(text = message)
    labelx.pack()

label = tk.Label(frame,text="Choose a Filtering Method ",bg="#3e646c")
label.pack()

button1 = tk.Button(frame, text="User Based Collaborative Filtering", command=userBased)
button1.pack(side=tk.LEFT)
button2 = tk.Button(frame, text="Item Based Collaborative Filtering",  command=itemBased)
button2.pack(side=tk.LEFT)
canvas = tk.Canvas(root, height = 500, width = 600, bg="#263d42")
canvas.pack()
global labelx

labelx = tk.Label(canvas,text="")
root.mainloop()
