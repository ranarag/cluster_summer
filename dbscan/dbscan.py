import sys
import math
import random
import subprocess
import matplotlib.pyplot as plt



def main():
	
	num_points = 150
	dimensions = 2
	lower = 0
	upper = 20
	num_clusters = 0
	opt_cutoff = 0.05
	file1 = open('input.txt','r')
	#num_points= int(file.readline())
	points = []
	for i in range(num_points):
		l = file1.readline().split(",")
		l.remove("\n")
		l = map(float,l)
		#l[0]=int(l[0])
		points.append(Point(l))
	#print str(points)
	#points = [makeRandomPoint(dimensions,lower,upper) for i in xrange(num_points)]
	file1.close()
	#file = open('results.txt', 'w')
	for i in range(1):
		eps = i/10.0
		minPts = i*2
		file1 = open('input.txt','r')
		points = []
		for i in range(num_points):
			l = file1.readline().split(",")
			l.remove("\n")
			l = map(float,l)
		
			points.append(Point(l))
	
		file1.close()
		clusters = dbscan(points,0.5,10)
		avg = dunn(len(clusters), clusters)
		avg1 = davies(len(clusters), clusters)
		s = "eps = "+str(eps)+ " minPts = "+str(minPts)+" dunn_index = "+str(avg)+" davies = "+str(avg1)+" number of clusters = "+str(len(clusters)) +"\n" 
		print s
		#file.write(s)
	#file.close()
		#avg += 
	j = 0
	for i,c in enumerate(clusters):
		x= []
		y =[]
		
		for p in c.points:
			j+=1
			#print "coords of point is = "+str(len(p.coords))
			print " Cluster: ", i, "\t  Point :", p
			#x.append(p.coords[0])
			#y.append(p.coords[1])
		#plt.plot(x,y,'ro')
		#plt.show()
	

		
	#avg /= 10
	print str(j)
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
		self.centroid = self.calculateCentroid()
		
	def __repr__(self):
	
		return str(self.points)
	
	def update(self, points):
		
		
		old_centroid = self.centroid
		self.points = points
		self.centroid = self.calculateCentroid()
		shift = getDistance(old_centroid, self.centroid)
		return shift
	
	def calculateCentroid(self):
		
		numPoints = len(self.points)
		coords = [p.coords for p in self.points]
		unzipped = zip(*coords)
		centroid_coords = [math.fsum(x)/numPoints for x in unzipped]
		return Point(centroid_coords)
	

def dbscan(points, eps, minPts):
	
	clusters = []
	i = 0
	counter = 0
	while len(points) > 1:
		#print "length of points=" + str(len(points))
		initial = random.sample(points, 1)
		
		points.remove(initial[0])
		lists,points = getNeighbours(points,initial[0],eps,minPts)
		if len(lists) >= minPts:
			clusters.append(Cluster(initial))
			clusters[i].update(lists)
			i += 1
		else:
			lists.remove(initial[0])
			
			for m in lists:
				points.append(m)
				#lists.remove(lists[i])

	for i in clusters:
		if len(i.points) < minPts:
			clusters.remove(i)
	return clusters
	
	
	



		
	
def getNeighbours(points, p,eps,minPts):
	lists = []
	lists.append(p)
	j = 0
	while len(points) >1:
		#print "length of points=" + str(len(points))
		if j >= len(lists):
			break
		mp = lists[j]
		for i in points:
			if getDistance(mp,i) <=eps:
				lists.append(i)
				points.remove(i)
		j += 1

	
	
	return lists,points


	
def getDistance(a, b):
	
		ret = reduce(lambda x,y: x + pow((a.coords[y]-b.coords[y]), 2),range(a.n),0.0)
		return math.sqrt(ret)

def makeRandomPoint(n, lower, upper):
	
	p = Point([random.uniform(lower, upper) for i in range(n)])
	return p

def dunn(k, clusters):

	if k <= 1:
		return 0.0
	distance = 200000.0
	initiald = distance
	for i in range(k):
		for j in range(i+1,k):
			new_dist=getClusterDistance(clusters[i],clusters[j])
			distance=min(distance, new_dist)
	
	dist2 = 1.0
	for i in range(k):
		numPoints = len(clusters[i].points)
		
		for j in clusters[i].points:
			for v in clusters[i].points:
				dist2=max(dist2, getDistance(j, v))

	dunn_index = distance/dist2		
	return dunn_index


def getClusterDistance(a, b):

	dist = 99999999999999999999.0
	for i in a.points:

		for j in b.points:

			dist = min(dist, getDistance(i, j))


	return dist

def davies(k, clusters):
	db = 0.0
	if k <= 1:
		return 1.0
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
			
			
				
		
	
		
