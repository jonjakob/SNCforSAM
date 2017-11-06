from scipy.spatial import distance
from scipy import optimize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import numpy as np
import libNearestNeighbor
import SNC

def snc(samples, labels):

	n = samples.shape[0]
	d = samples.shape[1]
	
	percent = 0.1
	numNewPoints = int(np.floor(n*percent))
	
	#f1 = SNC.obj1_1nn
	#fgr1 = SNC.optGammaGr_1nn
	#f2 = SNC.obj2_1nn
	#fgr2 = SNC.optPointsGr_1nn
	f1 = SNC.obj1_knn
	fgr1 = SNC.optGammaGr_knn
	f2 = SNC.obj2_knn
	fgr2 = SNC.optPointsGr_knn
	gamma_maxiter = 10
	points_maxiter = 50
	k = 5

	newSamples, newLabels = compress_fast(k,samples,labels,numNewPoints,f1,fgr1,f2,fgr2, gamma_maxiter, points_maxiter)

	return newSamples, newLabels




def compress(k,X,y,numNewPoints,f1,fgr1,f2,fgr2, gamma_maxiter, points_maxiter):
	
	d = X.shape[1]
	gammas = 2**np.array([-6,-4,-2,0,2,4,6], dtype='float32')
	
	V,vy = initialize(X, y, numNewPoints)

	objective = []
	optPoints = []
	for g in range(len(gammas)):
		x0 = np.asarray(gammas[g])
		# optGamma = optimize.fmin_bfgs(f1, x0, fprime=fgr1, maxiter=gamma_maxiter, args=(V, vy, X, y))
		# optV	 = optimize.fmin_bfgs(f2, V, fprime=fgr2, maxiter=points_maxiter, args=(vy, X, y, optGamma))
		optGamma = optimize.fmin_bfgs(f1, x0, fprime=fgr1, maxiter=gamma_maxiter, args=(V, vy, X, y, k))
		optV	 = optimize.fmin_bfgs(f2, V, fprime=fgr2, maxiter=points_maxiter, args=(vy, X, y, optGamma, k))
		optV 	 = optV.reshape(numNewPoints, d)

		neigh = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto', metric='euclidean', n_jobs=-1)
		neigh.fit(optV, vy)
		obj = neigh.score(X, y)

		objective.append(obj)
		optPoints.append(optV)

	idx = np.argmax(objective)
	optV = optPoints[idx]


	return optV, vy




def compress_fast(k,X,y,numNewPoints,f1,fgr1,f2,fgr2, gamma_maxiter, points_maxiter):
	
	d = X.shape[1]
	gamma = 0.0156
	
	V,vy = initialize(X, y, numNewPoints)

	x0 = np.asarray(gamma)
	#optGamma = optimize.fmin_bfgs(f1, x0, fprime=fgr1, maxiter=gamma_maxiter, args=(V, vy, X, y))
	#optV	 = optimize.fmin_bfgs(f2, V, fprime=fgr2, maxiter=points_maxiter, args=(vy, X, y, optGamma))
	optGamma = optimize.fmin_bfgs(f1, x0, fprime=fgr1, maxiter=gamma_maxiter, args=(V, vy, X, y, k))
	optV	 = optimize.fmin_bfgs(f2, V, fprime=fgr2, maxiter=points_maxiter, args=(vy, X, y, optGamma, k))
	optV 	 = optV.reshape(numNewPoints, d)


	return optV, vy




# ----- modified knn_functions including weighted knn

def obj1_knn(s, V, vy, X, y, k):

	n = X.shape[0]
	d = X.shape[1]	
	M = s * np.identity(d)

	nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(V)
	distances, indices = nbrs.kneighbors(X)
	ind = np.unique(indices, axis=0)
	points = V[ind]
	centroids = np.sum(points, axis=1) / k

	neigh = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto', metric='euclidean', n_jobs=-1)
	neigh.fit(V, vy)
	cy = neigh.predict(centroids)

	m = len(cy)
	Cs = np.dot(centroids,M)
	Xs = np.dot(X,M)

	K = np.exp(- distance.cdist(Xs, Cs, metric='sqeuclidean'))
	Z = np.sum(K, axis=1)
	P = np.multiply(np.transpose(K),(1/Z))
	logP = np.log(np.transpose(P))

	p = np.zeros(n)
	p2 = np.zeros(n)
	un = np.unique(y)

	for i in range(len(un)):
		jj = np.where(cy == un[i])[0]
		ii = np.where( y == un[i])[0]

		pis = np.amax(logP[ii,:], axis=1)

		a = logP[ii,:]
		b = a[:,jj]
		log_plus = pis + np.log(np.sum(np.exp(np.subtract(np.transpose(b), pis)), axis=0))
		p[ii] = np.exp(log_plus)

	obj = - sum(np.log(p))

	return obj




def obj2_knn(V0, vy, X, y, s, k):

	n = X.shape[0]
	d = X.shape[1]	
	M = s * np.identity(d)
	V = V0.reshape(len(vy),d)

	nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(V)
	distances, indices = nbrs.kneighbors(X)
	ind = np.unique(indices, axis=0)
	points = V[ind]
	centroids = np.sum(points, axis=1) / k

	neigh = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto', metric='euclidean', n_jobs=-1)
	neigh.fit(V, vy)
	cy = neigh.predict(centroids)

	m = len(cy)
	Cs = np.dot(centroids,M)
	Xs = np.dot(X,M)

	K = np.exp(- distance.cdist(Xs, Cs, metric='sqeuclidean') * 0.5)
	Z = np.sum(K, axis=1)
	P = np.multiply(np.transpose(K),(1/Z))
	logP = np.log(np.transpose(P))

	p = np.zeros(n)
	p2 = np.zeros(n)
	un = np.unique(y)

	for i in range(len(un)):
		jj = np.where(cy == un[i])[0]
		ii = np.where( y == un[i])[0]

		pis = np.amax(logP[ii,:], axis=1)

		a = logP[ii,:]
		b = a[:,jj]
		log_plus = pis + np.log(np.sum(np.exp(np.subtract(np.transpose(b), pis)), axis=0))
		p[ii] = np.exp(log_plus)

	obj = - sum(np.log(p))

	return obj




def optGammaGr_knn(s, V, vy, X, y, k):

	n = X.shape[0]
	d = X.shape[1]	
	M = s * np.identity(d)

	nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(V)
	distances, indices = nbrs.kneighbors(X)
	ind = np.unique(indices, axis=0)
	points = V[ind]
	centroids = np.sum(points, axis=1) / k

	neigh = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto', metric='euclidean', n_jobs=-1)
	neigh.fit(V, vy)
	cy = neigh.predict(centroids)

	m = len(cy)
	Cs = np.dot(centroids,M)
	Xs = np.dot(X,M)

	K = np.exp(- distance.cdist(Xs, Cs, metric='sqeuclidean'))
	Z = np.sum(K, axis=1)
	P = np.multiply(np.transpose(K),(1/Z))
	logP = np.log(np.transpose(P))

	p = np.zeros(n)
	p2 = np.zeros(n)
	un = np.unique(y)

	for i in range(len(un)):
		jj = np.where(cy == un[i])[0]
		ii = np.where( y == un[i])[0]

		pis = np.amax(logP[ii,:], axis=1)

		a = logP[ii,:]
		b = a[:,jj]
		log_plus = pis + np.log(np.sum(np.exp(np.subtract(np.transpose(b), pis)), axis=0))
		p[ii] = np.exp(log_plus)

	obj = - sum(np.log(p))

	Q0 = np.equal(np.tile(y.reshape(y.shape[0],1),(1,m)), np.tile(cy,(n,1))).astype(int)
	Q = np.subtract(np.transpose(Q0), p)
	inv_p = 1 / p

	T = - np.multiply(Q, P)
	Tt = np.transpose(T)

	DIST = distance.cdist(X, centroids, metric='sqeuclidean')
	DISTT = np.multiply(DIST,Tt)

	h = inv_p.reshape(inv_p.shape[0],1)
	F = np.multiply(DISTT, h)

	pisI = np.sum(F) * s

	gr = - 2 * pisI

	return gr




def optPointsGr_knn(V0, vy, X, y, s, k):

	n = X.shape[0]
	d = X.shape[1]	
	M = s * np.identity(d)
	V = V0.reshape(len(vy),d)

	nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(V)
	distances, indices = nbrs.kneighbors(X)
	ind = np.unique(indices, axis=0)
	points = V[ind]
	centroids = np.sum(points, axis=1) / k

	neigh = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto', metric='euclidean', n_jobs=-1)
	neigh.fit(V, vy)
	cy = neigh.predict(centroids)

	m = len(cy)
	Cs = np.dot(centroids,M)
	Xs = np.dot(X,M)

	K = np.exp(- distance.cdist(Xs, Cs, metric='sqeuclidean') * 0.5)		#multiplication with 0.5 not done in optGamma .. why?
	Z = np.sum(K, axis=1)
	P = np.multiply(np.transpose(K),(1/Z))
	logP = np.log(np.transpose(P))

	p = np.zeros(n)
	p2 = np.zeros(n)
	un = np.unique(y)

	for i in range(len(un)):
		jj = np.where(cy == un[i])[0]
		ii = np.where( y == un[i])[0]

		pis = np.amax(logP[ii,:], axis=1)

		a = logP[ii,:]
		b = a[:,jj]
		log_plus = pis + np.log(np.sum(np.exp(np.subtract(np.transpose(b), pis)), axis=0))
		p[ii] = np.exp(log_plus)

	obj = - sum(np.log(p))

	Q0 = np.equal(np.tile(y.reshape(y.shape[0],1),(1,m)), np.tile(cy,(n,1))).astype(int)
	Q = np.subtract(np.transpose(Q0), p)
	inv_p = 1 / p
	POP = np.multiply(P,inv_p)

	QPOP = np.multiply(Q, POP)
	QPOPt = np.transpose(QPOP)
	Xt = np.transpose(X)
	Ct = np.transpose(centroids)
	Mt = np.transpose(M)
	grC = np.dot(-Xt, QPOPt) + np.multiply(Ct, np.sum(QPOP, axis=1))
	grC = np.dot(np.dot(Mt, M), grC)

	grCt = np.transpose(grC)
	gr = np.zeros((len(vy), d))

	for j in range(len(vy)):
		tmpi = np.where(ind == j)[0]
		tmpp = grCt[tmpi]
		gr[j] = np.sum(tmpp, axis=0) / len(tmpp)	

	gr = np.transpose(gr)

	return gr.ravel(order='F')




# -------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----- original 1nn_functions from kusner

def obj1_1nn(s, V, vy, X, y):

	n = X.shape[0]
	d = X.shape[1]	
	m = len(vy)
	M = s * np.identity(d)

	Xs = np.dot(X,M)
	Vs = np.dot(V,M)

	K = np.exp(- distance.cdist(Xs, Vs, metric='sqeuclidean'))
	Z = np.sum(K, axis=1)
	P = np.multiply(np.transpose(K),(1/Z))
	logP = np.log(np.transpose(P))

	p = np.zeros(n)
	p2 = np.zeros(n)
	un = np.unique(y)

	for i in range(len(un)):
		jj = np.where(vy == un[i])[0]
		ii = np.where( y == un[i])[0]

		pis = np.amax(logP[ii,:], axis=1)

		a = logP[ii,:]
		b = a[:,jj]
		log_plus = pis + np.log(np.sum(np.exp(np.subtract(np.transpose(b), pis)), axis=0))
		p[ii] = np.exp(log_plus)

	obj = - sum(np.log(p))

	return obj




def obj2_1nn(V0, vy, X, y, s):

	n = X.shape[0]
	d = X.shape[1]	
	m = len(vy)
	M = s * np.identity(d)
	V = V0.reshape(m,d)

	Xs = np.dot(X,M)
	Vs = np.dot(V,M)

	K = np.exp(- distance.cdist(Xs, Vs, metric='sqeuclidean') * 0.5)
	Z = np.sum(K, axis=1)
	P = np.multiply(np.transpose(K),(1/Z))
	logP = np.log(np.transpose(P))

	p = np.zeros(n)
	p2 = np.zeros(n)
	un = np.unique(y)

	for i in range(len(un)):
		jj = np.where(vy == un[i])[0]
		ii = np.where( y == un[i])[0]

		pis = np.amax(logP[ii,:], axis=1)

		a = logP[ii,:]
		b = a[:,jj]
		log_plus = pis + np.log(np.sum(np.exp(np.subtract(np.transpose(b), pis)), axis=0))
		p[ii] = np.exp(log_plus)

	obj = - sum(np.log(p))

	return obj




def optGammaGr_1nn(s, V, vy, X, y):

	n = X.shape[0]
	d = X.shape[1]	
	m = len(vy)
	M = s * np.identity(d)

	Xs = np.dot(X,M)
	Vs = np.dot(V,M)

	K = np.exp(- distance.cdist(Xs, Vs, metric='sqeuclidean'))
	Z = np.sum(K, axis=1)
	P = np.multiply(np.transpose(K),(1/Z))
	logP = np.log(np.transpose(P))

	p = np.zeros(n)
	p2 = np.zeros(n)
	un = np.unique(y)

	for i in range(len(un)):
		jj = np.where(vy == un[i])[0]
		ii = np.where( y == un[i])[0]

		pis = np.amax(logP[ii,:], axis=1)

		a = logP[ii,:]
		b = a[:,jj]
		log_plus = pis + np.log(np.sum(np.exp(np.subtract(np.transpose(b), pis)), axis=0))
		p[ii] = np.exp(log_plus)

	obj = - sum(np.log(p))

	Q0 = np.equal(np.tile(y.reshape(y.shape[0],1),(1,m)), np.tile(vy,(n,1))).astype(int)
	Q = np.subtract(np.transpose(Q0), p)
	inv_p = 1 / p

	T = - np.multiply(Q, P)
	Tt = np.transpose(T)

	DIST = distance.cdist(X, V, metric='sqeuclidean')
	DISTT = np.multiply(DIST,Tt)

	h = inv_p.reshape(inv_p.shape[0],1)
	F = np.multiply(DISTT, h)

	pisI = np.sum(F) * s

	gr = - 2 * pisI

	return gr




def optPointsGr_1nn(V0, vy, X, y, s):

	n = X.shape[0]
	d = X.shape[1]	
	m = len(vy)
	M = s * np.identity(d)
	V = V0.reshape(m,d)

	Xs = np.dot(X,M)
	Vs = np.dot(V,M)

	K = np.exp(- distance.cdist(Xs, Vs, metric='sqeuclidean') * 0.5)		#multiplication with 0.5 not done in optGamma .. why?
	Z = np.sum(K, axis=1)
	P = np.multiply(np.transpose(K),(1/Z))
	logP = np.log(np.transpose(P))

	p = np.zeros(n)
	p2 = np.zeros(n)
	un = np.unique(y)

	for i in range(len(un)):
		jj = np.where(vy == un[i])[0]
		ii = np.where( y == un[i])[0]

		pis = np.amax(logP[ii,:], axis=1)

		a = logP[ii,:]
		b = a[:,jj]
		log_plus = pis + np.log(np.sum(np.exp(np.subtract(np.transpose(b), pis)), axis=0))
		p[ii] = np.exp(log_plus)

	obj = - sum(np.log(p))

	Q0 = np.equal(np.tile(y.reshape(y.shape[0],1),(1,m)), np.tile(vy,(n,1))).astype(int)
	Q = np.subtract(np.transpose(Q0), p)
	inv_p = 1 / p
	POP = np.multiply(P,inv_p)

	QPOP = np.multiply(Q, POP)
	QPOPt = np.transpose(QPOP)
	Xt = np.transpose(X)
	Vt = np.transpose(V)
	Mt = np.transpose(M)
	gr = np.dot(-Xt, QPOPt) + np.multiply(Vt, np.sum(QPOP, axis=1))
	gr = np.dot(np.dot(Mt, M), gr)

	return gr.ravel(order='F')




def initialize(samples, labels, numNewPoints):

	n = samples.shape[0]
	d = samples.shape[1]
    
	un = np.unique(labels)
	numClasses  = len(un)
	ratios = np.zeros(numClasses)

	for i in range(numClasses):
		ratios[i] = sum(labels == un[i])
    
	ratios = ratios / sum(ratios)
	counts = np.floor(ratios*numNewPoints)

	newPointsIdx = np.array([])
	for j in range(numClasses):
		if counts[j]:
			idx1 = np.where(labels==un[j])[0]
			idx2 = np.random.randint(len(idx1), size=int(counts[j]))
			smp = idx1[idx2]
			newPointsIdx = np.concatenate((newPointsIdx, smp), axis=0)			

	rest = numNewPoints - len(newPointsIdx)
	samps = np.random.randint(numClasses, size=rest)
	for k in samps:
		idx1 = np.where(labels==un[k])[0]
		idx2 = np.random.randint(len(idx1), size=1)
		smp = idx1[idx2]
		newPointsIdx = np.concatenate((newPointsIdx, smp), axis=0)
    
	V = samples[newPointsIdx.astype(int),:]
	vy = labels[newPointsIdx.astype(int)]
    
	return V, vy




