import csv
import numpy as np
csvName = "data.csv"

w, h = 5, 75;
DataMatrix = [["" for i in range(h)] for j in range(w)]

with open(csvName, 'r') as f:
	reader = csv.reader(f)
	for i, row in enumerate(reader):
		print(row)
		for j in range(h):
			#print((i, j))
			DataMatrix[i][j]=row[j]
		if i == 4:
			break
DataMatrix = np.array(DataMatrix)
print(DataMatrix)