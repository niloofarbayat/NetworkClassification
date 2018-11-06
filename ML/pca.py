import numpy as np
import sklearn.ensemble, sklearn.model_selection
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def float_list(l):
	return [float(x)  for x in l]

def read_csv(file_path, has_header=True):
	with open(file_path) as f:
		if has_header: f.readline()
		data = []
		for line in f:
			line = line.strip().split(",")
			data.append([x for x in line])
	return data

filename = '/Users/wjackson/Downloads/Weston_Jackson_ancestry_composition_0.5.csv'
dataset=[]
dataset+=read_csv(filename)

print(len(dataset))

result = []
X = np.array([float_list(z[3:5]) for z in dataset])
Y = np.array([z[0] for z in dataset])

lookupTable, indexed_dataSet = np.unique(Y, return_inverse=True)

#y = np.array([float_list(z[0]) for z in dataset])

print(np.shape(X))
plt.scatter(X[:,0], X[:,1], c=indexed_dataSet, s=5)
plt.show()

"""
fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig)
pca = PCA(n_components=3)
pca.fit(X)
T = pca.transform(X)
ax.scatter(T[0], T[1], T[2], , s=2)
plt.show()
"""