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

filename = ''
dataset=[]
dataset+=read_csv(filename)

result = []
X = np.array([float_list(z[1:25]) for z in dataset])
y = np.array([float_list(z[0]) for z in dataset])

fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig)
pca = PCA(n_components=3)
pca.fit(X)
T = pca.transform(X)
ax.scatter(T[0:2000, 0], T[0:2000, 1], T[0:2000, 2], c=y[0:2000], s=2)
plt.show()