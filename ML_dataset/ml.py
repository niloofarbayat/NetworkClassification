import numpy as np
import sklearn.ensemble, sklearn.model_selection
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_csv(file_path, has_header=True):
	with open(file_path) as f:
		if has_header: f.readline()
		data = []
		for line in f:
			line = line.strip().split(",")
			data.append([x for x in line])
	return data
def float_list(l):
	return [float(x)  for x in l]

nf_fname = 'netflix_2s.csv'
yt_fname = 'youtube_2s.csv'
browsing_fname = 'browsing_2s.csv'
testRounds=50
max_features="auto"
max_depth=None

def getfit(X_train, y_train):
	rf = sklearn.ensemble.RandomForestClassifier(
		max_depth=max_depth, 
		max_features=max_features)
	rf.fit(X_train, y_train)
	return rf

def main():
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

	for test in range(testRounds):
		skf = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True)
		for train_index, test_index in skf.split(X, y):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			#print X_train.shape, y_train.shape
			model = getfit(X_train, y_train)
			l1 = model.predict(X_test)
			print 1. * np.sum(l1 == y_test) / len(y_test)
			
			tw, tn, ty, fw, fn, fy= 0, 0, 0, 0, 0, 0
			for i in range(len(y_test)):
				if y_test[i]==0:
					tw+=1
					if l1[i]!=0: 
						fw+=1
				if y_test[i]==1:
					tn+=1
					if l1[i]!=1: 
						fn+=1
				if y_test[i]==2:
					ty+=1
					if l1[i]!=2: 
						fy+=1

			print 1 - 1. * fw/tw, 1 - 1. * fn/tn, 1 - 1. * fy/ty
		#break
	return 0,0

if __name__ == "__main__":
	task = 2
	for max_depth in range(1,11):
		main()
		#yt = train_and_test(yt_fname,nf_fname)
		#nf = train_and_test(nf_fname, yt_fname)
		#print max_depth, all, web, yt, nf
