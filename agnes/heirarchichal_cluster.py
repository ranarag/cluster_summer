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
	file = open('input.txt','r')
	#num_points= int(file.readline())
	points = []
	for i in range(num_points):

		l = file.readline().split(",")
		l.remove('\n')
		l = map(float,l)
		#l[0]=int(l[0])
		points.append(Point(l))
	#print str(points)
	#points = [makeRandomPoint(dimensions,lower,upper) for i in xrange(num_points)]
	
	

	clusters = agnes(points)
	#avg = davies(len(clusters), clusters)
		#avg += 
	
	for m in clusters:
		file = open(str(m.d)+'.txt','w')
		print " Cluster: ", m
		file.write(str(m)+'\n')
		for i,c in enumerate(m.clusters):
			file.write(str(c)+'\n')
			for p in c.points:
			
			#print "coords of point is = "+str(len(p.coords))
				print  "\t  Point :", p
				file.write(str(p)+'\n')

		file.close()		
			#plt.plot(x,y,'ro')
			#plt.show()
	

		
	#avg /= 10
	#print str(j)
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
		
	
	def calculateCentroid(self):
		
		numPoints = len(self.points)
		coords = [p.coords for p in self.points]
		unzipped = zip(*coords)
		centroid_coords = [math.fsum(x)/numPoints for x in unzipped]
		return Point(centroid_coords)

class Clusters(Cluster):

	def __init__(self,clusters,d):

		self.clusters = clusters
		self.k = len(clusters)
		self.d = d
		
	def __repr__(self):

		return "d = "+str(self.d)+" k = "+str(self.k) + " clusters = "+str(self.clusters)


def agnes(points):
	
	lists = [Cluster([p]) for p in points]
	d = 0
	clusters = []
	clusters.append(Clusters(lists,d))
	
	while len(lists) > 1:
		d += 1
		
		
		for i in lists:
			
			lists.remove(i)
			
			for j in lists:
				#print " i = "+str(i)+" j = "+str(j)
				
					
				if getDistance(i.centroid,j.centroid) <= d:
					l = []
					for p in j.points:
						l.append(p)
					for p in i.points:
						l.append(p)
					
					i.update(l)
					
					lists.remove(j)
					
				

			lists.append(i)
		clusters.append(Clusters(lists,d))
		
		#print " i = "+str(i)

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

	dist = 99999999999999999999999999999999999999
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

#def rmsstd()
			
	
if __name__=="__main__":
	main()
			
			
				
		
	
		
