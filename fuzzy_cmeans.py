import sys
import math
import random
import subprocess
import matplotlib.pyplot as plt


def main():
	
	num_points = 100
	dimensions = 2
	lower = 0
	upper = 200
	num_clusters = 4
	opt_cutoff = 0.05
	
	points = [makeRandomPoint(dimensions, lower, upper) for i in xrange(num_points)]
	
	clusters = kmeans(points, num_clusters, opt_cutoff)
	
	for i,c in enumerate(clusters):
		for p in c.points:
			print " Cluster: ", i, "\t Point :", p
	davies(num_clusters, clusters)
	
class Point:
	
	def __init__(self, coords):
		
		self.coords = coords
		self.n = len(coords)
		
	
	def __repr__(self):
	
		return str(self.coords)
	

class Cluster:
	
	def __init__(self, points):
		
		self.points = points
		self.n = points[0].n
		self.centroid = self.calculateCentroid(points,clusters)
		
	def __repr__(self):
	
		return str(self.points)
	
	def update(self, points,tot_points,clusters,m):
		
		
		old_centroid = self.centroid
		self.points = points
		self.centroid = self.calculateCentroid(tot_points,clusters,m)
		shift = getDistance(old_centroid, self.centroid)
		return shift
	
	def calculateCentroid(self,points,clusters,m):
		mew_ij = 0.0
		xmew_ij = 0.0
		for i in points:

			value = pow(getmembership(clusters,i,m,self.centroid),m)
			mew_ij += value
			xmew_ij += value*i

		return xmew_ij/mew_ij
		
	

def kmeans(points, k, cutoff):
	
	initial = random.sample(points, k)
	clusters = [Cluster([p]) for p in initial]
	counter = 0
	
	while True:
		
		lists = [[] for c in clusters]
		clus_count = len(clusters)
		counter+=1
		for p in points:
			
			smallest_distance = getDistance(p, clusters[0].centroid)
			clusterIndex = 0
			
			for i in range(clus_count-1):
				
				distance = getDistance(p, clusters[i+1].centroid)
				if distance < smallest_distance:
					
					smallest_distance = distance
					clusterIndex = i+1
				
			
			lists[clusterIndex].append(p)
		biggest_shift = 0.0
			
		for i in range(clus_count):
				
			shift = clusters[i].update(lists[i])
			if biggest_shift < shift:
					
				biggest_shift = shift
				
		if biggest_shift < cutoff:
				
			break
		
	return clusters
		
	
def getDistance(a, b):
	
		ret = reduce(lambda x,y: x + pow((a.coords[y]-b.coords[y]), 2),range(a.n),0.0)
		return math.sqrt(ret)

def makeRandomPoint(n, lower, upper):
	
	p = Point([random.uniform(lower, upper) for i in range(n)])
	return p

def dunn(k, clusters):

	distance = 200
	for i in range(k):
		for j in range(i+1,k):
			new_dist=get_Cluster_distance(clusters[i],clusters[j])
			distance=min(distance, new_dist)

	for i in range(k):
		numPoints = len(clusters[i].points)
		dist2 = 0
		for j in clusters[i].points:
			for v in clusters[i].points:
				dist2=max(dist2, getDistance(j, v))

	dunn_index = distance/dist2		
	print str(dunn_index)


def get_cluster_Distance(a, b):

	dist = 0.0
	for i in a.points:

		for j in b.points:

			dist = min(dist, getDistance(i, j))


	return dist

def davies(k, clusters):
	db = 0.0
	for i in range(k):
		
		s_i = 0.0

		for p in clusters [i].points:
			
			s_i += getDistance(p, clusters[i].centroid)
		s_i = s_i/len(clusters[i].points)
		r_i = 0.0
		for j in range(i+1,k):

			d_ij=getDistance(clusters[i].centroid,clusters[j].centroid)
			s_j = 0.0

			for p in clusters[j].points:
				
				s_j += getDistance(p, clusters[j].centroid)
			s_j = s_j/len(clusters[j].points)

			r_i = max(r_i, ((s_i + s_j)/d_ij))

		db += r_i

	db = db/k

	print str(db)

def getmembership(clusters, point,m,center):

	d_ij = getDistance(point, center)
	sum = 0.0
	for i in clusters:
		d_ik = getDistance(point,i.centroid)
		sum += pow((d_ij/d_ik),((2/m) - 1))
	return 1/sum
			
def calcCost()		
	
if __name__=="__main__":
	main()