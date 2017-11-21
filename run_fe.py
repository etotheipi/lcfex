#! /usr/bin/python
import argparse
import csv
from fe import FeatureExtractor, ClassificationFunc
from rest_api import LcApiCall, GetLoanListing
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read, parse and run FE on csv file")
    parser.add_argument(dest="trainFile",  help="Path of .csv file to calc stats")
    #parser.add_argument(dest="predictFile",  help="Path of .csv file to run predictions")
    args = parser.parse_args()

    trainCsv = csv.DictReader(open(args.trainFile, 'r'))
    fexTrain = FeatureExtractor()
    fexTrain.RunFeatureExtractor(trainCsv, context='train')
    cols, dataMtrx = fexTrain.GetFexMatrix()
    allCols, allMtrx = fexTrain.GetAllDataListOfLists()
    avgs, stds, boolList = fexTrain.GetColumnStats()
    ratios, totalCount, sumClass, globalAvg = fexTrain.RunBinaryStats()
    scoredLoans = fexTrain.ScoreLoans(gradeMin='C3')

    allLoans = GetLoanListing(getAllAvail=True)
    #predictCsv = csv.DictReader(open(args.predictFile, 'r'))
    fexPredict = FeatureExtractor()
    fexPredict.RunFeatureExtractor(allLoans, context='predict')
    # fexPredict now has a an allDataMtrx that we score using fexTrain
    scoredLoans = fexTrain.ScoreLoans(fexPredict.allDataMatrix, outFile='predict_scores.csv', gradeMin='C3')

    """
    print '*'*100
    print 'Example Features:'
    for k,v in zip(cols,dataMtrx[0,:]):
        print ' ', k.ljust(30), v

    print '*'*100
    print 'Example ALL Data:'
    for k,v in zip(allCols,allMtrx[0][:]):
        print ' ', k.ljust(30), v
    """

    ccoef = np.corrcoef(dataMtrx.T)
    
    with open('corrcoef.csv','w') as f:
        def WRITE(*args):
            f.write(', '.join(args) + '\n') 

        WRITE(' '*40, ',', ', '.join(cols))
        for i,row in enumerate(ccoef):
            WRITE(cols[i].ljust(40), ',', ', '.join(['%0.3f'%v for v in row]))

        WRITE('')

        WRITE('AVG,,'.ljust(40), ', '.join(['%0.3f'%v for v in avgs]))
        WRITE('STDEV,,'.ljust(40), ', '.join(['%0.3f'%v for v in stds]))

        WRITE('ClassRatio,,'.ljust(40), ', '.join(['%0.3f'%v for v in ratios]))
        WRITE('ClassSum,,'.ljust(40), ', '.join(['%0.3f'%v for v in totalCount]))
        WRITE('TotalSum,,'.ljust(40), ', '.join(['%0.3f'%v for v in sumClass]))
        WRITE('')
        WRITE('GlobalAvg,%0.3f' % globalAvg)




