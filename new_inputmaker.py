import sys
from random import*
import math
import matplotlib.pyplot as plt


def main():
	file = open('input.txt','w')
	file1 = open('new_input.txt','r')
	for i in range(150):
		s = file1.readline().replace("Iris-setosa","")
		s = s.replace("Iris-versicolor","")
		s = s.replace("Iris-virginica","")
		file.write(s)

	file.close()
	file1.close()
if __name__ == '__main__':
	main()


