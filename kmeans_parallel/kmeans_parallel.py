import sys
import math
import random
import subprocess
import csv
import matplotlib.pyplot as plt


def main():
	
	num_points = 1000
	dimensions = 2
	lower = 0
	upper = 200
	num_clusters = 4
	opt_cutoff = 0.05
	oversample = 1
	'''points = [makeRandomPoint(dimensions, lower, upper) for i in xrange(num_points)]
	
	clusters = kmeans(points, num_clusters, opt_cutoff)'''
	'''file = open('input.txt','r')
	#num_points= int(file.readline())
	points = []
	num_points = 150
	for i in range(num_points):

		s = file.readline().replace(" ","")
		l = s.split(",")
		l.remove("\n")
		l = map(float,l)
		
		points.append(Point(l))
	clusters = kmeans(points, num_clusters, opt_cutoff)
	for i,c in enumerate(clusters):
		for p in c.points:
			print " Cluster: ", i, "\t Point :", p
	davies(num_clusters, clusters)'''
	file1 = open("results_wine1.csv",'w')
	fo = csv.writer(file1,lineterminator='\n')
	fo.writerow([' dunn_index','davies','number of clusters'])
	x = []
	y = []
	z = []
	
	for i in range(20):
		
		avg = 0.0
		avg1 = 0.0
		for j in range(3):
			file = open('1.csv','r')
			s=file.readline()
			points = []
			num_points = 4580
			for k in range(num_points):

				l = file.readline().split(",")
				#l.remove(l[0])
				l = map(float,l)
				points.append(Point(l))
			file.close()	
			clusters = kmeans(points, i+1, opt_cutoff)
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
	plt.savefig("dunn")
	
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
	

def kmeans(points, k, cutoff):
	
	initial = cluster_initialize(points, k,2)
	
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

	if k <= 1:
		return 0.0
	distance = 20000000.0
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

	dist = 9999999999999999.0
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


def cluster_initialize(points, noc, l):

	initial = random.sample(points, 1)
	C = []

	points.remove(initial[0])
	lp = len(points)
	C.append(initial[0])
	dist = 0
	for p in points:
		dist +=getDistance(initial[0], p)
	n = int(math.ceil(math.log(dist)))
	for i in range(n):
		psi = []
		for j in range(lp):
			mini = 200
			for k in range(len(C)):
				if mini > getDistance(points[j], C[k]):
					mini = getDistance(points[j],C[k])

			psi.append(mini)

		phi_c = math.fsum(psi)
		for j in range(lp):
			p_x = l*psi[j]/phi_c
			if p_x >= random.uniform(0,1):
				C.append(points[i])
				points.remove(points[i])
				lp -= 1

	w = [0 for i in range(len(C))]
	for i in range(lp):
		mini = getDistance(points[i], C[0])
		index = 0
		for j in range(1,len(C)):
			if mini > getDistance(points[i], C[j]):
				mini = getDistance(points[i],C[j])
				index = j
		w[index] += 1

	for i in range(len(C)):
		swapped = False
		for j in range(i+1,len(C)):
			if w[j] < w[j-1]:
				tmp = w[j]
				w[j] = w[j-1]
				w[j-1] = tmp
				tmp = C[j]
				C[j] = C[j-1]
				C[j-1] = tmp
				swapped = True

		if not swapped:
			break

	lists = []
	#print str(noc)
	for i in range(noc):
		lists.append(C[n-1-i])
	#print str(len(lists))
	return lists






	






	
	
if __name__=="__main__":
	main()
			
			
				
		
	
		
