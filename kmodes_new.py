import sys
import math
import random
import subprocess
import matplotlib.pyplot as plt


def main():
	
	num_points = 1000
	dimensions = 2
	lower = 0
	upper = 200000
	num_clusters = 2
	opt_cutoff = 0.05
	file = open('input1.txt','r')
	#num_points= int(file.readline())
	points = []
	for i in range(num_points):

		l = file.readline().replace('\n',"")
		l = list(l)
		#l.remove('\n')
		#l = map(float,l)
		#l[0]=int(l[0])
		points.append(Point(l))
	#print str(points)'''
	#points = [makeRandomPoint(num_points,lower,upper) for i in xrange(num_points)]
	avg = 0.0
	clusters = kmeans(points, num_clusters, opt_cutoff)
	#avg += davies(num_clusters, clusters)
	'''for i in range(10):

		clusters = kmeans(points, num_clusters, opt_cutoff)
		avg += davies(num_clusters, clusters)'''
	j = 0
	for i,c in enumerate(clusters):
		x= []
		y =[]
		#j += 1
		for p in c.points:
			j += 1
			print " Cluster: ", i, "\t Point :", p
			#x.append(p.coords[0])
			#y.append(p.coords[1])
		#plt.plot(x,y,'ro')
		#plt.show()

	print str(j)	
	#avg /= 10
	print str(avg)

	
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
		self.centroid = self.calculateMode()
		
	def __repr__(self):
	
		return str(self.points)
	
	def update(self):
		
		
		old_centroid = self.centroid
		#self.points = points
		self.centroid = self.calculateMode()
		shift = getDistance(old_centroid, self.centroid)
		
	
	def calculateMode(self):
		
		numPoints = len(self.points)
		coords = [p.coords for p in self.points]
		unzipped = zip(*coords)
		
		centroid_coords = [modethm(x) for x in unzipped]
		return Point(centroid_coords)
	
def modethm(x):
	k = list(x)
	k.sort()
	element = k[0]
	maxfreq = 1
	freq = 1
	for i in range(1, len(k)):
		if k[i-1] == k[i]:
			freq +=1
		else :
			freq = 0
		if maxfreq < freq:
			maxfreq = freq
			element = k[i]

	return element

def kmeans(points, k, cutoff):
	
	initial = random.sample(points, k)
	clusters = [Cluster([p]) for p in initial]
	counter = 0
	for p in initial:
		points.remove(p)
	while True:
		
		lists = [[] for c in clusters]
		clus_count = len(clusters)
		counter+=1
		for p in points:
			
			smallest_distance = getDistance2(p, clusters[0])
			clusterIndex = 0
			
			for i in range(clus_count-1):
				
				distance = getDistance2(p, clusters[i+1])
				if distance < smallest_distance:
					
					smallest_distance = distance
					clusterIndex = i+1
				
			
			#lists[clusterIndex].append(p)
			clusters[i+1].points.append(p)
			clusters[i+1].update()
		break
	shift = 1
	while shift != 0:
		shift = 0		
		for i in range(clus_count):
			for p in clusters[i].points:
				smallest_distance = getDistance2(p,clusters[i])
				clusterIndex = i
				clusters[i].points.remove(p)
				for j in range(clus_count):
					if i ==j:
						continue
					distance = getDistance2(p,clusters[j])
					if distance < smallest_distance:
						smallest_distance =distance
						clusterIndex = j
						shift = 1

				clusters[clusterIndex].points.append(p)
				clusters[i].update()
				clusters[clusterIndex].update()
				

			
				
		
		
	return clusters
		
def getDistance2(point, cluster):

	dist = 0.0
	n_l = len(cluster.points)
	for i in range(min(point.n,len(cluster.centroid.coords))):
		if point.coords[i] != cluster.centroid.coords[i]:
			dist +=1
		else:
			n_j,n_l = getFrequency(i,point.coords[i],cluster)
			k = n_j/n_l
			dist += 1 - k

	return dist

def getFrequency(attribute,value, cluster):

	n_j = 0
	n_l = 0
	for i in cluster.points:
		if attribute >= len(i.coords):
			continue
		if value == i.coords[attribute]:
			n_j += 1
		n_l += 1

	if n_l == 0:
		return 0,1
	return n_j,n_l



def getDistance(a, b):
	
	dist = 0
	for i in range(min(len(a.coords),len(b.coords))):
		if a.coords[i] != b.coords[i]:
			dist += 1

	return dist 


def makeRandomPoint(n, lower, upper):
	
	p = Point([random.uniform(lower, upper) for i in range(n)])
	return p

def dunn(k, clusters):

	distance = 200000
	for i in range(k):
		for j in range(i+1,k):
			new_dist=getClusterDistance(clusters[i],clusters[j])
			distance=min(distance, new_dist)

	for i in range(k):
		numPoints = len(clusters[i].points)
		dist2 = 0
		for j in clusters[i].points:
			for v in clusters[i].points:
				dist2=max(dist2, getDistance(j, v))

	dunn_index = distance/dist2		
	print str(dunn_index)


def getClusterDistance(a, b):

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

	return db


			
	
if __name__=="__main__":
	main()
			
			
				
		
	
		
