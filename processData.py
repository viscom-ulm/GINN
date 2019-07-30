'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file processData.py

    \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import math
import os
import numpy as np
from os import listdir
from os.path import isfile, join

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'MCCNN/utils'))

from PyUtils import visualize_progress

def process_model(modelPath):
    if not isfile(modelPath[:-4]+".npy"):
        corrupt = False
        fileDataArray = []
        with open(modelPath, 'r') as modelFile:        
            it = 0
            for line in modelFile:
                line = line.replace("\n", "")
                if ("nan" not in line) and ("inf" not in line):
                    currPoint = line.split(',')
                    try:
                        fileDataArray.append([float(currVal) for currVal in currPoint])
                    except ValueError:
                        corrupt = True
                    it+=1

        if not(corrupt):
            np.save(modelPath[:-4]+".npy", np.array(fileDataArray))
    
        return corrupt
    return False

def get_files():
    fileList = []
    dataFolders = ["ao_data", "gi_data", "sss_data"]
    datasets = ["training", "evaluation", "test"]
    for currDataFolder in dataFolders:
        for currDataSet in datasets:
            for f in listdir(currDataFolder+"/"+currDataSet) :
                if isfile(join(currDataFolder+"/"+currDataSet+"/", f)) and f.endswith(".txt"):
                    fileList.append(join(currDataFolder+"/"+currDataSet+"/", f))
    return fileList


if __name__ == '__main__':
    
    fileList = get_files()
    iter = 0
    maxSize = 0.0
    corruptFiles = []
    for inFile in fileList:
        if process_model(inFile):
            corruptFiles.append(inFile)
        if iter%100==0:
            visualize_progress(iter, len(fileList))
        iter = iter + 1
    for currCorruptFile in corruptFiles:
        print(currCorruptFile)
    print(len(corruptFiles))