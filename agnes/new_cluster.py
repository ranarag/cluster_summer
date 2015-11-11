import sys
import math
import random
import subprocess
import matplotlib.pyplot as plt
import csv

def main():
	
	num_points = 1000
	dimensions = 2
	lower = 0
	upper = 200
	num_clusters = 18
	opt_cutoff = 0.05
	file = open('spiral.txt','r')
	#file = open('headpose_l.txt','r')
	#num_points= int(file.readline())
	points = []
	num_points = 312
	s = file.readline()
	for i in range(num_points):

		s = file.readline().replace('\n','')
		l = s.split("\t")
		l.remove(l[2])
		l = map(float,l)
		#print str(l)
		
		#l[0]=int(l[0])
		points.append(Point(l))
	#points = [makeRandomPoint(dimensions, lower, upper) for i in xrange(num_points)]
	
	clusters = kmeans(points, num_clusters, opt_cutoff)
	
	for i,c in enumerate(clusters):
		x = []
		y = []
		for p in c.points:
			print " Cluster: ", i, "\t Point :", p
			x.append(p.coords[0])
			y.append(p.coords[1])
		plt.plot(x,y,'ro')
		plt.show()
	
	
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
		self.centroid = self.calculateCentroid()
		self.compactness = None
	def __repr__(self):
	
		return str(self.points)
	
	def update(self, points):
		
		
		old_centroid = self.centroid
		self.points = points
		self.centroid = self.calculateCentroid()
		#shift = getDistance(old_centroid, self.centroid)
		#return shift
	def checkfeasibility(self,point):

		if self.clusterlen() == 1:
			self.compactness = self.calculatecompactness(point)/2.0
			self.points.append(point)
			self.update(self.points)
			return 1 

		else:
			old_compactness = self.compactness
			new_compactness = (self.calculatecompactness(point)+(old_compactness*self.clusterlen()))/(self.clusterlen()+1)
			if new_compactness > old_compactness:
				return 0
			else:
				self.compactness = new_compactness
				self.points.append(point)
				self.update(self.points)
				return 1

	def calculatecompactness(self,point):
		r = 0.0
		for i in self.points:
			r += getAbsDiff(i,point)

		
		return r



	def calculateCentroid(self):
		
		numPoints = len(self.points)
		coords = [p.coords for p in self.points]
		unzipped = zip(*coords)
		centroid_coords = [math.fsum(x)/numPoints for x in unzipped]
		return Point(centroid_coords)
	
	def clusterlen(self):
		return len(self.points)



def merge(a,b):
	#compactness calculation
	if a.clusterlen() == 1:
		com = 0.0
	else:
		com = a.compactness*a.clusterlen()

	if b.clusterlen() > 1:
		com += b.compactness*b.clusterlen()

	#print str(a.points)+ "    " +str(b.points)
	for p in b.points:
		com += a.calculatecompactness(p)
	com =com /(a.clusterlen()+b.clusterlen())
	#compactness calculation finished
	a.compactness = com
	a.update(a.points+b.points)
	return a

def kmeans(points, k, cutoff):
	
	initial , points = cluster_initialize(points, k)
	
	clusters = [Cluster([p]) for p in initial]
	counter = 0
	
	for i in range(len(points)):
		smallest_distance = getDistance(clusters[0].centroid,points[i])
		flag = 0
		clusterIndex = 0
		for j in range(len(clusters)-1):
			dist = getDistance(clusters[j+1].centroid,points[i])
			if dist < smallest_distance:
				smallest_distance = dist
				clusterIndex = j+1
		clen = len(clusters)
		for j in range(clen):
			if j >= clen:
				break
			dist = getDistance(clusters[j].centroid,points[i])
			if dist == smallest_distance and clusterIndex != j:
				#merging clusters
				flag = 1
				clusters[clusterIndex] = merge(clusters[clusterIndex],clusters[j])
				clusters.remove(clusters[j])
				clen -= 1
				if j >= clen:
					break
				#merging done

		if flag == 0:
			#if not merged go thorugh basic protocol
			if clusters[clusterIndex].checkfeasibility(points[i]) == 0:
				clusters.append(Cluster([points[i]]))
		else:
			# if merged then take the poit inside clusterIndex wala cluster
			old = clusters[clusterIndex].compactness
			clusters[clusterIndex].compactness = (clusters[clusterIndex].calculatecompactness(points[i])+(old*clusters[clusterIndex].clusterlen()))/(clusters[clusterIndex].clusterlen()+1)
			clusters[clusterIndex].points.append(points[i])
			clusters[clusterIndex].update(clusters[clusterIndex].points)

	k = len(clusters)
	clusters2 = clusters
	prev_dunn = dunn(k,clusters)
	print str(prev_dunn)
	distance = 999999.0
	for i in range(k):
		for j in range(i+1,k):
			new_dist=getClusterDistance(clusters[i],clusters[j])
			distance=min(distance, new_dist)

	for i in range(k):
		if k == 1:
			break
		for j in range(i+1,k):
			if k == j:
				break 

			new_dist=getClusterDistance(clusters[i],clusters[j])
			if new_dist <= distance:
				for p in clusters[i].points:
					clusters[j].points.append(p)
				clusters[i].update(clusters[j].points)
				clusters.remove(clusters[j])
				k = k-1
	new_dunn = dunn(k,clusters)
	print str(new_dunn)
	m = 0
	while prev_dunn < new_dunn:
		k = len(clusters)
		clusters2 = clusters
		prev_dunn = new_dunn
		distance = 999999.0
		for i in range(k):
			if k <= i:
				break
			for j in range(i+1,k):
				if k <= j:
					break
				new_dist=getClusterDistance(clusters[i],clusters[j])
				distance=min(distance, new_dist)

		for i in range(k):
			if k <= i:
				break
			for j in range(i+1,k):
				if k <= j:
					break 

				new_dist=getClusterDistance(clusters[i],clusters[j])
				if new_dist <= distance:
					for p in clusters[i].points:
						clusters[j].points.append(p)
					clusters[i].update(clusters[j].points)
					clusters.remove(clusters[j])
					k = k-1
		new_dunn = dunn(k,clusters)
		#print str(new_dunn)
		m += 1
		#if new_dunn > prev_dunn:
			#prev_dunn = new_dunn
			#clusters2 = clusters
	print str(m)
	return clusters2



		
	
def getDistance(a, b):
	
		ret = reduce(lambda x,y: x + pow((a.coords[y]-b.coords[y]), 2),range(a.n),0.0)
		return math.sqrt(ret)
def getAbsDiff(a,b):
	ret = 0.0
	for i in range(a.n):
		ret += math.fabs(a.coords[i]- b.coords[i])
	return ret
def makeRandomPoint(n, lower, upper):
	
	p = Point([random.uniform(lower, upper) for i in range(n)])
	return p

def dunn(k, clusters):
	if k <= 1:
		return 0.0

	distance = 999999.0
	for i in range(k):
		

		for j in range(i+1,k):
			new_dist=getClusterDistance(clusters[i],clusters[j])
			distance=min(distance, new_dist)
	dist2 = 1.0
	for i in range(k):
		numPoints = len(clusters[i].points)
		#print "ok "+str(numPoints)
		for j in clusters[i].points:
			for v in clusters[i].points:
				dist2=max(dist2, getDistance(j, v))

	dunn_index = distance/dist2		
	return dunn_index


def getClusterDistance(a, b):

	
	dist = 9999999999.0
	for i in a.points:

		for j in b.points:
			if getDistance(i,j) == 0:
				continue
			dist = min(dist, getDistance(i, j))


	return dist

def davies(k, clusters):
	if k <= 1:
		return 1.0
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

def cluster_initialize(points, k):

	initial = random.sample(points, 1)
	lists = []
	#print str(initial)
	lists.append(initial[0])
	points.remove(initial[0])
	#print str(lists[0].coords)
	while True:
		if len(lists) == k:
			break
		for l in lists:
			maxd = getDistance(l,points[0])
			maxp = points[0]
			for p in points:
				dist = getDistance(l,p)
				if dist > maxd:
					maxd = dist
					maxp = p

			lists.append(maxp)
			points.remove(maxp)
			if len(lists) == k:
				break

	return lists, points



		



	
	
if __name__=="__main__":
	main()
			
			
				
		
	
		
