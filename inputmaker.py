import sys
from random import*
import math
import matplotlib.pyplot as plt

def randcoords(file):

	
	n =200
	file.write(str(n)+'\n')
	x1 = []
	y1 = []
	i=25

	for k in range(1,101):
		if k%25 == 0:
			i = k
		x = randint(200*(i-1),200*i)
		y = randint(200*(i-1),200*i)

		x1.append(x)
		y1.append(y)
		
		file.write(str(x)+' '+str(y)+'\n')
	plt.plot(x1,y1,'ro')
	plt.show()
	#file.close()
def circle(file):
	r = 4
	rk = 8
	#file = open('input.txt','w')
	#n =100
	x1 = []
	y1 = []
	n =250
	file.write(str(n)+'\n')
	#file.write(str(n)+'\n')

	for i in range(51):
		x = -4.0000 +(0.16000000*i) 
		y = float(math.sqrt(16 - pow(x,2)))
		x1.append(x)
		y1.append(y)
		file.write(str(x)+' '+str(y)+'\n')
		x1.append(x)
		y1.append(-y)
		file.write(str(x)+' '+str(-y)+'\n')
	for i in range(201):
		x = -8.0000 +(0.08*i)
		y = math.sqrt(64 - pow(x,2))
		x1.append(x)
		y1.append(y)

		file.write(str(x)+' '+str(y)+'\n')
		x1.append(x)
		y1.append(-y)
		file.write(str(x)+' '+str(-y)+'\n')
	plt.plot(x1,y1,'ro')
	plt.show()
	

def main():
	file = open('input2.txt','w')
	#randcoords(file)
	circle(file)
	file.close()
if __name__ == '__main__':
	main()


