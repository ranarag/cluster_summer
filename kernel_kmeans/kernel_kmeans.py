import sys
import math
import random
import subprocess
import matplotlib.pyplot as plt
import csv

def main():
	
	num_points = 100
	dimensions = 2
	lower = 0
	upper = 20000
	num_clusters = 3
	opt_cutoff = 0.05
	
	file = open('spiral.txt','r')
	num_points= 312
	points = []
	l=file.readline()
	for i in range(num_points):

		#l = map(float,file.readline().split(" "))
		l = file.readline().split('\t')
		#l.remove('\n')
		v = l[2]
		#l[1]=l[1].replace('\n','')
		#print v[0]
		l.remove(l[2])
		print str(l)
		l.append(v[0])
		l = map(float, l)
		
		points.append(Point(l,i))
	
	#print str(points)
	#points = [makeRandomPoint(dimensions, lower, upper,i) for i in xrange(num_points)]
	avg = 0.0
	j=0
	for i in range(1):
	
		clusters = kernel_kmeans(points, num_clusters, opt_cutoff,14.8,0)
		avg += davies(num_clusters, clusters)
		j = 0
		for i,c in enumerate(clusters):
			
			x = []
			y = []
			for p in c.points:
				j +=1
				print " Cluster: ", i, "\t Point :", p
				x.append(p.coords[0])
				y.append(p.coords[1])
			plt.plot(x,y,'ro')
			plt.show()
	#avg /= 2
	print str(j)
	'''file1 = open("results_1_sigmoid.csv",'w')
	fo = csv.writer(file1,lineterminator='\n')
	fo.writerow([' dunn_index','davies','number of clusters'])
	x = []
	y = []
	z = []'''
	
	'''for i in range(2):
		
		avg = 0.0
		avg1 = 0.0
		for j in range(3):
			file = open('1.csv','r')
	
			points = []
			num_points = 480
			s = file.readline()
			for k in range(num_points):

				l = file.readline().split(",")
				#l.remove('')
				l = map(float,l)
				points.append(Point(l,k))
			file.close()	
			#clusters = kmedoid(points, i+1, opt_cutoff)
			clusters = kernel_kmeans(points, i+1, opt_cutoff,3.3,0)
			avg += dunn(i+1, clusters)
			avg1 += davies(i+1, clusters)
		avg /= 3
		avg1 /= 3
		lis = []
		lis.append(avg)
		lis.append(avg1)
		lis.append(len(clusters))
		s = " dunn_index = "+str(avg)+" davies = "+str(avg1)+" number of clusters = "+str(len(clusters)) +"\n" 
		print s
		x.append(len(clusters))
		y.append(avg)
		z.append(avg1)
		fo.writerow(lis)
	file1.close()
	plt.plot(x,y,'r--',x,z)
	plt.savefig("dunn_sigmoid")'''
	
class Point:
	
	def __init__(self, coords, i):
		
		self.coords = coords
		self.n = len(coords)
		self.index = i
	
	def __repr__(self):
	
		return str(self.coords) 
	

class Cluster:
	
	def __init__(self, points):
		
		self.points = points
		self.n = points[0].n
		self.centroid = self.calculateCentroid()
		
	def __repr__(self):
	
		return str(self.points)
	
	def update(self, points,kernel_g):
		
		
		old_centroid = self.centroid
		self.points = points
		self.centroid = self.calculateCentroid()
		#shift = fspdist(kernel_g,old_centroid, self.centroid)
		#shift = kernel_g[old_centroid.index][self.centroid.index]
		shift = getDistance(old_centroid,self.centroid)
		return shift
	
	def calculateCentroid(self):
		
		numPoints = len(self.points)
		coords = [p.coords for p in self.points]
		unzipped = zip(*coords)
		centroid_coords = [math.fsum(x)/numPoints for x in unzipped]
		#return Point(centroid_coords,numPoints)
		p = Point(centroid_coords,numPoints)
		dist = getDistance(self.points[0],p)
		minp = self.points[0]

		for i in self.points:
			ndist = getDistance(p,i)
			if ndist < dist:
				dist = ndist
				minp = i
		return minp
def kernel_kmeans(points, k, cutoff,gamma,delta):

	clusters = []
	kernel_g = gaussian_kernel(points,gamma)
	for i in range(1,k+1):

		initial = random.sample(points, 1)
		while Cluster(initial) in clusters:
			initial = random.sample(points, 1)
		clusters.append(Cluster(initial))
		print "kernel ok\n"
		clusters = kmeans(kernel_g,clusters,points,i,cutoff)
	return clusters
def kmeans(kernel_g,clusters,points, k, cutoff):
	'''kernel_g = gaussian_kernel(points,gamma)
	initial = random.sample(points, k)
	clusters = [Cluster([p]) for p in initial]'''
	counter = 0
	
	while True:
		
		lists = [[] for c in clusters]
		clus_count = len(clusters)
		counter+=1
		for p in points:
			
			smallest_distance = fspdist(kernel_g,p, clusters[0])
			clusterIndex = 0
			
			for i in range(clus_count-1):
				
				distance = fspdist(kernel_g,p, clusters[i+1])
				if distance < smallest_distance:
					
					smallest_distance = distance
					clusterIndex = i+1
				
			
			lists[clusterIndex].append(p)
		biggest_shift = 0.0
			
		for i in range(clus_count):
				
			shift = clusters[i].update(lists[i],kernel_g)
			if biggest_shift < shift:
					
				biggest_shift = shift
				
		if biggest_shift < cutoff:
				
			break
	print "kmeans ok\n"
	return clusters
		
	
def getDistance(a, b):
	
		ret = reduce(lambda x,y: x + pow((a.coords[y]-b.coords[y]), 2),range(min(a.n,b.n)),0.0)
		return math.sqrt(ret)

def makeRandomPoint(n, lower, upper,index):
	
	p = Point([random.uniform(lower, upper) for i in range(n)],index)
	return p

def dunn(k, clusters):
	if k <= 1:
		return 0.0

	distance = 200000.0
	for i in range(k):
		for j in range(i+1,k):
			new_dist=getClusterDistance(clusters[i],clusters[j])
			distance=min(distance, new_dist)
	dist2 = 0.0
	for i in range(k):
		numPoints = len(clusters[i].points)
		
		for j in clusters[i].points:
			for v in clusters[i].points:
				dist2=max(dist2, getDistance(j, v))

	dunn_index = distance/dist2		
	return dunn_index


def getClusterDistance(a, b):

	dist = 2000000000.0
	for i in a.points:

		for j in b.points:

			dist = min(dist, getDistance(i, j))


	return dist

def davies(k, clusters):
	if k <= 1:
		return 1.0
	db = 0.0
	for i in range(k):
		
		s_i = 0.0
		flag = 0
		for p in clusters [i].points:
			flag =1
			s_i += getDistance(p, clusters[i].centroid)
		#s_i = pow(s_i,float(1/4.0))
		s_i = s_i/len(clusters[i].points)
		r_i = 0.0
		for j in range(i+1,k):

			d_ij=getDistance(clusters[i].centroid,clusters[j].centroid)
			s_j = 0.0
			flag = 0
			for p in clusters[j].points:
				flag = 1
				s_j += getDistance(p, clusters[j].centroid)
			#s_j = pow(s_j,float(1/4.0))
			s_j = s_j/len(clusters[j].points)

			r_i = max(r_i, ((s_i + s_j)/d_ij))

		db += r_i

	db = db/k

	return db

def getVectorProduct(a,b):
	return reduce(lambda x,y: x +(a.coords[y]*b.coords[y]),range(a.n),0.0)

def gaussian_kernel(points,sigma):
	kernel_g = [[0 for x in range(len(points))] for x in range(len(points))]
	for i in range(len(points)):
		for j in range(len(points)):
			kernel_g[points[i].index][points[j].index] = math.exp(-pow(getDistance(points[i],points[j]),2)/(pow(sigma,2))*2)
	return kernel_g

def polynomial_kernel(points,gamma,delta):
	kernel_p = [[0 for x in range(len(points))] for x in range(len(points))]
	for i in range(len(points)):
		for j in range(len(points)):
			kernel_p[points[i].index][points[j].index] = pow(getVectorProduct(points[i],points[j])+gamma,delta)
	return kernel_p

def sigmoid_kernel(points,gamma,theta):
	kernel_s = [[0 for x in range(len(points))] for x in range(len(points))]
	for i in range(len(points)):
		for j in range(len(points)):
			kernel_s[points[i].index][points[j].index] = math.tanh((getVectorProduct(points[i],points[j])*gamma)+theta)
	return kernel_s

def fspdist(kernel_g,a,cluster):
	result = kernel_g[a.index][a.index]
	b = 0.0
	for i in cluster.points:
		b += kernel_g[a.index][i.index]
	b /=len(cluster.points)
	b *=2
	c = 0.0
	for i in cluster.points: 
		for j in cluster.points:
			c += kernel_g[i.index][j.index]

	c /=pow(len(cluster.points), 2)

	return result - b + c

def new_gaussian_kernel_dist(a,b,sigma):
	ret = math.exp(-pow(getDistance(a,b),2)/(pow(sigma,2))*2)
	return ret
	
if __name__=="__main__":
	main()
			
			
				
		
	
		
