import numpy as np
import sklearn.ensemble, sklearn.model_selection
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def float_list(l):
	return [float(x)  for x in l] #if x<1400 else 1400.0

def read_csv(file_path, has_header=True):
	with open(file_path) as f:
		if has_header: f.readline()
		data = []
		for line in f:
			line = line.strip().split(",")
			data.append([x for x in line])
	return data

nf_fname = 'netflix_2s.csv'
yt_fname = 'youtube_2s.csv'
browsing_fname = 'browsing_2s.csv'

dataset=[]
dataset+=read_csv(yt_fname)
y=[2]*len(dataset)
dataset+=read_csv(nf_fname)
y+=[1]*(len(dataset) - len(y))
	
dataset+=read_csv(browsing_fname)
y+=[0]*(len(dataset) - len(y))

result = []
X = np.array([ float_list(z[3:27]) for z in dataset])
y = np.array(y)

fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig)
pca = PCA(n_components=3)
pca.fit(X)
T = pca.transform(X)
ax.scatter(T[0:2000, 0], T[0:2000, 1], T[0:2000, 2], c=y[0:2000], s=2)
plt.show()