#! /usr/bin/python
import argparse
import csv
from fe import FeatureExtractor, ClassificationFunc
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read, parse and run FE on csv file")
    parser.add_argument(dest="csvFile",    help="Path of .csv file")
    args = parser.parse_args()

    theCsv = csv.DictReader(open(args.csvFile, 'r'))

    """
    #next(theCsv)
    #cols = next(theCsv)
    #indexLookup = dict([(c,i) for i,c in enumerate(cols)])
    #_ = lambda colName: indexLookup[colName]
            
    #print indexLookup
    #a = next(theCsv)
    #for k,v in zip(cols, a):
        #print k.ljust(20), v
    #hugeMatrix = run_feature_extract(theCsv, indexLookup)
    """

    #row = next(theCsv)
    #print row
    #out = FeatureExtractor.RunFeatureExtractor_Single(row)
    #print out
    #out['_CLASS_'] = ClassificationFunc(row)
    #for k in sorted(out.keys()):
        #print k.ljust(22), out[k]

    fex = FeatureExtractor()
    cols,dataMtrx, allCols,allMtrx = fex.RunFeatureExtractor(theCsv)
    normMtrx, avgs, stds = fex.RunNormalizer()
    ratios, totalCount, sumClass, globalAvg = fex.RunBinaryStats()

    print '*'*100
    print 'Example Features:'
    for k,v in zip(cols,dataMtrx[0,:]):
        print ' ', k.ljust(30), v

    print '*'*100
    print 'Example ALL Data:'
    for k,v in zip(allCols,allMtrx[0][:]):
        print ' ', k.ljust(30), v

    print '*'*100
    print 'Example NORMALIZED Data:'
    for k,v in zip(allCols, normMtrx[0][:]):
        print ' ', k.ljust(30), v



    ccoef = np.corrcoef(dataMtrx.T)
    
    with open('corrcoef.csv','w') as f:
        def WRITE(*args):
            f.write(', '.join(args) + '\n') 
            print ', '.join(args)

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

