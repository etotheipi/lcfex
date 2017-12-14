#! /usr/bin/python
import argparse
import csv
from fe import *
from rest_api import *
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read, parse and run FE on csv file")
    parser.add_argument(dest="trainFile",  help="Path of .csv file to calc stats")
    parser.add_argument(dest="predictFile",  help="Path of .csv file to run predictions")
    parser.add_argument('-t', '--threshold', dest='scoreThresh', type=float, default=0.0,
                        help='Pick out notes above certain score')
    parser.add_argument('-x', '--execute-trades', dest="executePrice", type=int,
                        help="Purchase new notes for this price that exceed the -t threshold")
    parser.add_argument('-p', '--portfolio', dest="addToPortfolio", 
                        help="Add purchased notes to portfolio with this ID or name")
    args = parser.parse_args()

    trainCsv = csv.DictReader(open(args.trainFile, 'r'))
    fexTrain = FeatureExtractor()
    fexTrain.RunFeatureExtractor(trainCsv, context='train')
    cols, dataMtrx = fexTrain.GetFexMatrix()
    allCols, allMtrx = fexTrain.GetAllDataListOfLists()
    avgs, stds, boolList = fexTrain.GetColumnStats()
    ratios, totalCount, sumClass, globalAvg = fexTrain.RunBinaryStats()
    scoredLoans = fexTrain.ScoreLoans(gradeMin='C5', 
                                      gradeMax='E2')

    scoredLoans = fexTrain.ScoreLoans(gradeMin='C5', 
                                      gradeMax='E2', 
                                      doRandom=True, 
                                      outFile='randomized.csv')

    # This is applying the same rules to that latest download of borrower data
    allLoans = GetLoanListing(getAllAvail=True)
    alreadyInvested = GetAlreadyInvestedIds()
    fexPredict = FeatureExtractor()
    fexPredict.RunFeatureExtractor(allLoans, context='predict')
    colNames,scoreMtrx = fexTrain.ScoreLoans(fexPredict.allDataMatrix, 
                                             outFile='predict_scores.csv', 
                                             gradeMin='C5',
                                             gradeMax='E2',
                                             alreadySet=alreadyInvested)

    # This does backtesting of the rules on an another dataset
    predictCsv = csv.DictReader(open(args.predictFile, 'r'))
    fexPredict = FeatureExtractor()
    fexPredict.RunFeatureExtractor(predictCsv, context='predict')
    backtest = fexTrain.ScoreLoans(fexPredict.allDataMatrix, 
                                   outFile='backtest_scores.csv', 
                                   gradeMin='E1')

    testRandom = fexTrain.ScoreLoans(fexPredict.allDataMatrix, 
                                     gradeMin='C5', 
                                     gradeMax='E2', 
                                     doRandom=True, 
                                     outFile='randomized_backtest.csv')

    if args.executePrice is None:
        doExec = False
        execPrice = 0
    else:
        doExec = True
        execPrice = args.executePrice

    if not execPrice % 25 == 0:
        raise Exception('Note investment price must be a multiple of $25')

    idList = GetNewLoanIdsAboveThresh(colNames, scoreMtrx, args.scoreThresh)

    CreateInvestOrderPayload([(lid, execPrice) for lid in idList], args.addToPortfolio)
