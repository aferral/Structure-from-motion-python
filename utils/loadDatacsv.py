import numpy as np

def load(csvFilename):
    with open(csvFilename,'r') as f:
        rawText = f.read()
        data = rawText.split('\n')[1:] #Saca el header en elemento 0
        rows = []
        for textRow in data:
            tempRow = []
            for elem in textRow.split(','):
                if elem != '':
                    tempRow.append(float(elem))
            if len(tempRow) != 0:
                rows.append(tempRow)
        matrix = np.array(rows)
        return matrix



if __name__ == '__main__':
    load("data/descriptLocI.csv")