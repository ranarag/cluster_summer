import sys
import math
import random
import subprocess
import matplotlib.pyplot as plt


def main():
	
	num_points = 100
	dimensions = 2
	lower = 0
	upper = 200000
	num_clusters = 30
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
	clusters = kernel_kmeans(points, num_clusters, opt_cutoff)
	dunn(num_clusters,clusters)
	#avg += davies(num_clusters, clusters)
	'''for i in range(10):

		clusters = kmeans(points, num_clusters, opt_cutoff)
		avg += davies(num_clusters, clusters)'''
	j = 0
	for i,c in enumerate(clusters):
		x= []
		y =[]
		#j += 1
		file=open(str(i)+".txt",'w')
		for p in c.points:
			j += 1
			#print " Cluster: ", i, "\t Point :", p
			s = ''.join(p.coords)
			file.write(str(p.coords)+"\n")
			#x.append(p.coords[0])
			#y.append(p.coords[1])
		#plt.plot(x,y,'ro')
		#plt.show()
		file.close()

	#print str(j)	
	#avg /= 10
	#print str(avg)

	
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
		#coords.sort(lent)
		length = []
		centroid_coords = []
		for i in coords:
			if len(i) not in length:
				length.append(len(i))
		#print str(length)
		length.sort()
		x = 0
		for i in length:
			new_coords = []
			for j in coords:
				if i == len(j):
					new_coords.append(j)
					coords.remove(j)
			unzipped = zip(*new_coords)
			while x < i: 
				centroid_coords.append(modethm(unzipped[x]))

				x +=1
			#print str(centroid_coords)
		return Point(centroid_coords)
def lent(a,b):
	if len(b) > len(a):
		return 1
	return 0

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

def kernel_kmeans(points, k, cutoff):

	clusters = []
	initial = random.sample(points, 1)
	clusters.append(Cluster(initial))
	points.remove(initial[0])
	clusters = kmeans(clusters,points,1,cutoff)
	for i in range(2,k+1):
		

		initial = random.sample(clusters[0].points, 1)
		clusters.append(Cluster(initial))
		#print "initial =" +str(initial)
		clusters[0].points.remove(initial[0])
		clusters[0].update()
		#print "kernel ok\n"
		clusters = kmeans(clusters,None,i,cutoff)
		#points.append(initial[0])

	return clusters

def kmeans(clusters,points, k, cutoff):
	
	'''initial = random.sample(points, k)
	clusters = [Cluster([p]) for p in initial]'''
	counter = 0
	'''for p in initial:
		points.remove(p)'''
	while points is not None:		
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
			clusters[clusterIndex].points.append(p)
			clusters[clusterIndex].update()
		break
	shift = 1
	while shift != 0:
		shift = 0		
		for i in range(k):
			for p in clusters[i].points:
				smallest_distance = getDistance2(p,clusters[i])
				clusterIndex = i
				clusters[i].points.remove(p)
				for j in range(k):
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
	
	dist = 0.0
	for i in range(min(len(a.coords),len(b.coords))):
		if a.coords[i] != b.coords[i]:
			dist += 1.0

	return dist 
def getDistance3(a, b):
	
	dist = 0.0
	for i in range(min(len(a.coords),len(b.coords))):
		if a.coords[i] != b.coords[i]:
			dist += 1

	return dist +abs(len(a.coords)-len(b.coords))

def makeRandomPoint(n, lower, upper):
	
	p = Point([random.uniform(lower, upper) for i in range(n)])
	return p

def dunn(k, clusters):

	distance = 9999999.0
	for i in range(k):
		for j in range(i+1,k):
			new_dist=getClusterDistance(clusters[i],clusters[j])
			distance=min(distance, new_dist)
			
	dist2 = 0.0
	for i in range(k):
		
		
		for j in clusters[i].points:
			for v in clusters[i].points:
				ndist = getDistance3(j, v)
				dist2=max(dist2, ndist)
				
	
	dunn_index = distance/dist2		
	print str(dunn_index)


def getClusterDistance(a, b):

	dist = 99999999999.0
	for i in a.points:

		for j in b.points:

			dist = min(dist, getDistance3(i, j))


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
			
			
				
		
	
		
