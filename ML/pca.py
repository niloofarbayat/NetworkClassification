import numpy as np
import sklearn.ensemble, sklearn.model_selection
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
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

filename = 'training/GCDay1stats.csv'
dataset=[]
dataset+=read_csv(filename)

print(len(dataset))

result = []
X = np.array([float_list(z[1:42]) for z in dataset])
y = np.array([z[0] for z in dataset])

print("Entering filtering section! ")
snis, counts = np.unique(y, return_counts=True)
above_min_conns = list()

for i in range(len(counts)):
    if counts[i] > 250:
        above_min_conns.append(snis[i])

print("Filtering done. SNI classes remaining: ", len(above_min_conns))
indices = np.isin(y, above_min_conns)
X = X[indices]
y = y[indices]

lookupTable, indexed_dataSet = np.unique(y, return_inverse=True)

print("Filtered shape of X =", np.shape(X))
print("Filtered shape of y =", np.shape(y))  

fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig)
pca = PCA(n_components=3)
pca.fit(X)
T = pca.transform(X)


colors = ['red','green','blue','purple','yellow','black','brown']

for i, u in enumerate(lookupTable):
    print(i,u)
    xi = T[y == u]
    yi = y[y == u]
    ax.scatter(xi[:,0], xi[:,1], xi[:,2], c=colors[i], label=str(u), s=2)
ax.set_zticklabels([])
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.legend()
plt.show()
