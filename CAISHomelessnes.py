import csv
import numpy as np
csvName = "data.csv"

w, h = 5, 41;
DataMatrix = [["" for i in range(h)] for j in range(w)]

with open(csvName, 'r') as f:
	reader = csv.reader(f)
	for i, row in enumerate(reader):
		#print(row)
		counter = 0;
		if i != 0:
			for j in range(len(row)):
				#print((i, j))
				if j == 1 or j == 3 or j == 5 or j == 6 or j == 8 or j == 9 or j == 10 or j == 11 or j == 12 or j == 15 or j == 16 or j == 18 or j == 20 or j == 21 or j == 23 or j == 24 or j == 26 or j == 28 or j == 30 or j == 32 or j == 33 or j == 34  or j == 35 or j == 36 or j == 37 or j == 39 or j == 40 or j == 42 or j == 43 or j == 44 or j == 45 or j == 48 or j == 49 or j == 50 or j == 52 or j == 55 or j == 56 or j == 58 or j == 72 or j == 73 or j == 74:
					DataMatrix[i-1][counter]=row[j]
					counter += 1
		if i == w:
			break
DataMatrix = np.array(DataMatrix)
pregnancyColumn = DataMatrix[:, 24]
pregnancyColumn[pregnancyColumn == ""] = "m"

petColumn = DataMatrix[:, 40]
petColumn[petColumn == ""] = "n"

LGBTQQI2Column = DataMatrix[:, 39]
LGBTQQI2Column[LGBTQQI2Column == ""] = "n"

print(DataMatrix)

