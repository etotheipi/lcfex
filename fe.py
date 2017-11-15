#! /usr/bin/python
import csv
from datetime import datetime
from dateutil import parser as dateParser
import os
import numpy as np

HOME_OWNERSHIPS = [ \
                    'rent', 
                    'own', 
                    'mortgage',
                    'other'
                 ]

PURPOSES = [ \
                'debt_consolidation', 
                'credit_card', 
                'small_business',
                'home_improvement',
                'major_purchase',
                #'vacation',
                #'moving',
                #'car',
                #'house',
                #'medical',
                #'other',
                #'wedding',
                #'renewable_energy'
           ]


####################################################################################################
# A singleton class that tracks all the feature extraction methods and applies them
####################################################################################################
class FeatureExtractor(object):
    FeatureList = []
    FilterList = []
    OthersList = []

    @staticmethod
    def AddFeature(funcObj, featureNames):
        FeatureExtractor.FeatureList.append([funcObj, featureNames])

    @staticmethod
    def AddFilter(funcObj):
        FeatureExtractor.FilterList.append(funcObj)

    @staticmethod
    def AddOther(funcObj, valueNames):
        FeatureExtractor.OthersList.append([funcObj, valueNames])

    def __init__(self):
        self.fexMatrix = None
        self.colNames = []
        self.allDataMatrix = None
        self.allDataColNames = []
        self.normalizedMtrx = None
        self.numRows = 0
        self.numFeatures = 0
        self.binaryList = []

    def RunFeatureExtractor_Single(self, row, scoringFunction=None):
        # This checks all the filters in order, skipping this row if any fail
        for fn in FeatureExtractor.FilterList:
            if not fn(row):
                return

        # Run all the fex functions 
        nList,vList = [],[]
        for func,names in FeatureExtractor.FeatureList:
            nList.extend(names)
            vList.extend(func(row))

        nList.append('_CLASS_')
        vList.append(ClassificationFunc(row))

        self.numFeatures = len(vList)

        # Make a non-numeric matrix of identical rows, but with extra cols to identify rows
        nListAll,vListAll = nList[:],vList[:]
        for func,names in FeatureExtractor.OthersList:
            nListAll.extend(names)
            vListAll.extend(func(row))

        if self.fexMatrix is None:
            self.numRows = 0
            self.fexMatrix = np.ndarray((10000, len(nList)), dtype=float)
            self.colNames = nList[:]
            self.allDataMatrix = [ [0]*len(nListAll) for i in range(10000) ]
            self.allDataColNames = nListAll[:]

        if self.numRows >= self.fexMatrix.shape[0]:
            self.fexMatrix.resize(self.fexMatrix.shape[0]+10000, self.numFeatures)
            self.allDataMatrix.extend([ [0]*len(nListAll) for i in range(10000) ])

        self.fexMatrix[self.numRows] = vList
        self.allDataMatrix[self.numRows] = vListAll
        self.numRows += 1


    def RunFeatureExtractor(self, rowIterator):
        # Don't know fexMatrix numColumns until the first row is calculated, done in the _single call
        self.fexMatrix = None

        for i,row in enumerate(rowIterator):
            self.RunFeatureExtractor_Single(row)

        self.fexMatrix.resize(self.numRows, self.numFeatures)
        self.allDataMatrix = self.allDataMatrix[:self.numRows]
        
        return self.colNames, self.fexMatrix, self.allDataColNames, self.allDataMatrix

    
    def RunBinaryStats(self):
        self.allClassSum = 0
        self.featureRowCount = [0]*self.numFeatures
        self.featureClassSum = [0]*self.numFeatures
        for row in self.fexMatrix:
            self.allClassSum += row[-1]
            for c in range(self.numFeatures):
                if self.binaryList[c]:
                    if row[c]==1:
                        self.featureClassSum[c] += row[-1]
                        self.featureRowCount[c] += 1
                
        self.ratioList = [0] * self.numFeatures
        for c in range(self.numFeatures):
            if self.featureRowCount[c] > 0:
                self.ratioList[c] = float(self.featureClassSum[c]) / float(self.featureRowCount[c])
        
        self.totalAvg = float(self.allClassSum) / float(self.numRows)
        return self.ratioList, self.featureClassSum, self.featureRowCount, self.totalAvg


    def RunNormalizer(self):
        self.avgVec = np.mean(self.fexMatrix, axis=0)
        self.stdVec = np.std(self.fexMatrix, axis=0)
        self.binaryList = [True] * self.numFeatures

        self.normalizedMtrx = [ [0]*len(self.allDataColNames) for i in range(self.numRows) ]
    
        for row in self.allDataMatrix:
            for c in range(self.numFeatures):
                if row[c] not in [0,1]:
                    self.binaryList[c] = False
            

        for r,row in enumerate(self.allDataMatrix):
            for c in range(len(self.allDataColNames)):
                if c >= self.numFeatures or self.stdVec[c]==0 or self.binaryList[c]:
                    self.normalizedMtrx[r][c] = row[c]
                else:
                    self.normalizedMtrx[r][c] = (row[c] - self.avgVec[c]) / self.stdVec[c]

        return self.normalizedMtrx, self.avgVec, self.stdVec

    def ScoreFunction(self, row):
        pass
        

    def RunScoring(self, row):
        pass
        


##### DECORATOR #####
def FexMethod(*featureNames):
    def outerWrapper(func):
        def newFunc(*args, **kwargs):
            # Wrapped function will always return a list, all lists will be concatenated at end
            out = func(*args, **kwargs)
            outList = list(out) if isinstance(out, (list,tuple)) else [out]
            for i in range(len(outList)):
                if isinstance(outList[i], bool):
                    outList[i] = 1 if outList[i] else 0

            if not len(featureNames) == len(outList):
                raise Exception('Feature names list not same size as output: Names:%d, Out:%d' % \
                                (len(featureNames), len(outList)))
            return outList

        FeatureExtractor.AddFeature(newFunc, featureNames)
    return outerWrapper

##### DECORATOR #####
# I really should've combined this logic with the FexMethod, but I was too lazy
def OtherDataMethod(*otherNames):
    def outerWrapper(func):
        def newFunc(*args, **kwargs):
            # Wrapped function will always return a list, all lists will be concatenated at end
            out = func(*args, **kwargs)
            outList = list(out) if isinstance(out, (list,tuple)) else [out]
            for i in range(len(outList)):
                if isinstance(outList[i], bool):
                    outList[i] = 1 if outList[i] else 0

            if not len(otherNames) == len(outList):
                raise Exception('Other names list not same size as output: Names:%d, Out:%d' % \
                                (len(otherNames), len(outList)))
            return outList

        FeatureExtractor.AddOther(newFunc, otherNames)
    return outerWrapper

##### DECORATOR #####
def FilterMethod(func):
    FeatureExtractor.AddFilter(func)
    return func


def GRADE2VALUE(gradeOrSubgrade):
    major = ord(gradeOrSubgrade[0].lower()) - ord('a')
    minor = 0 if len(gradeOrSubgrade)==1 else (int(gradeOrSubgrade[1]) - 1)
    return major + minor*0.2

def BitListFromSet(strValue, listOfOptions):
    out = [0] * len(listOfOptions)
    if strValue.lower() in listOfOptions:
        idx = listOfOptions.index(strValue.lower())
        out[idx] = 1
    return out
    

####################################################################################################
def ClassificationFunc(row):
    return 1 if row['loan_status'].lower() == 'fully paid' else 0
####################################################################################################


####################################################################################################
# Filtering Methods
####################################################################################################
@FilterMethod
def filter_completed(row):
    return row['loan_status'].lower() in ['fully paid', 'charged off']

#@FilterMethod
#def filter_verified_income_only(row):
    #return not row['verification_status'].lower().strip() == 'not verified'

@FilterMethod
def filter_grade_b2d_only(row):
    return GRADE2VALUE('B2') <= GRADE2VALUE(row['sub_grade']) < GRADE2VALUE('F')

@FilterMethod
def filter_annual_inc_gt_30k(row):
    return float(row['annual_inc']) >= 40000

@FilterMethod
def filter_has_revolving_util_value(row):
    return '%' in row['revol_util']

####################################################################################################
# Feature Extraction Methods
####################################################################################################
@FexMethod('GradeFloat')
def fex_lc_grade_float(row):
    # A: [0,1), B: [1,2), C: [2,3), D: [3,4), E: [4,5), F: [5,6)
    return GRADE2VALUE(row['sub_grade'])


@FexMethod('MonthlyIncome2Pmt')
def fex_install_per_income(row):
    moPayment = float(row['installment'])
    moIncome  = float(row['annual_inc'])/12
    return moIncome/moPayment


@FexMethod(*['HomeOwn_'+t for t in HOME_OWNERSHIPS])
def fex_home(row):
    return BitListFromSet(row['home_ownership'], HOME_OWNERSHIPS)


@FexMethod(*['Purpose_'+t for t in PURPOSES])
def fexmulti_purpose(row):
    return BitListFromSet(row['purpose'], PURPOSES)

@FexMethod('Debt2Income')
def fex_debt2income(row):
    return float(row['dti'])

@FexMethod('Delinq2yrs')
def fex_delinq_2yrs(row):
    return float(row['delinq_2yrs'])

@FexMethod('OldestCreditYrs')
def fex_oldest_credit_line(row):
    dtOrig = dateParser.parse(row['earliest_cr_line'])
    dtNow  = datetime.now()
    return (dtNow - dtOrig).days / 365.

@FexMethod('VerifyStatus')
def fex_verify_stat(row):
    vstr = row['verification_status'].lower()
    if 'not' in vstr:
        return 0
    elif 'source' in vstr:
        return 2 
    else:
        return 1 

@FexMethod('NumOpenAccts')
def fex_open_accts(row):
    return float(row['open_acc'])

@FexMethod('NumAcctsEver')
def fex_ever_accts(row):
    return float(row['total_acc'])

@FexMethod('RevolvUtil')
def fex_revolveutil(row):
    return float(row['revol_util'].rstrip('%')) / 100.

@FexMethod('Revolv2Income')
def fex_openaccts(row):
    revBal = float(row['revol_bal'].rstrip('%')) / 100.
    annualInc = float(row['annual_inc'])
    return revBal/annualInc

@FexMethod('TitleCapNone')
def fex_titlecapnone(row):
    return row['title'].lower() == row['title']

@FexMethod('TitleCapNoneOrAll')
def fex_titlecapnoneallcaps(row):
    t = row['title']
    return t in [t.lower(), t.upper()]

@FexMethod('TitleCapNoneOrAllOrNoTitle')
def fex_titlecapnoneallnone(row):
    t = row['title']
    return t in [t.lower(), t.upper()] or len(t.strip())<2

@FexMethod('HasDescr')
def fex_buyerdescription_nonzero(row):
    return len(row['desc'].strip()) == 0 or row['desc']==row['desc'].lower()

@FexMethod('DescrLength')
def fex_buyerdescription_nonzero(row):
    desc = row['desc'].strip()
    if len(desc) > 0 and '>' in desc:
        desc = '>'.join(desc.split('>')[1:])
    if desc.endswith('<br>'):
        desc = desc[:-4]

    # Kind of a logarithmic func...
    l = len(desc)
    if l == 0:    return 0
    elif l < 20:  return 1
    elif l < 30:  return 2
    elif l < 50:  return 3
    elif l < 80:  return 4
    elif l < 120: return 5
    elif l < 200: return 6
    elif l < 400: return 7
    else:         return 8
    return len(desc)

@FexMethod('EmployeeLeader')
def fex_like_mgr_empl(row):
    posList = ['supervisor','manager','vp', 'admin', 'senior', 'officer', 'lead', 'director', 'president']
    return any([pos in row['emp_title'].lower() for pos in posList])

@FexMethod('EmployeeMedical')
def fex_like_med_empl(row):
    posList = ['nurse', 'drug', 'doctor', 'resident', 'phys', 'medical', 'medicine', 'hospital', 'ologist', 'ometrist', 'registered']
    titleLc = row['emp_title'].lower()
    if titleLc.startswith('rn') or titleLc.endswith(' rn'):   
        return 1

    if 'sales' in titleLc:
        return 0

    return any([pos in row['emp_title'].lower() for pos in posList])

@FexMethod('EmployeeTech')
def fex_like_tech_empl(row):
    posList = ['engineer', 'tech', 'info', 'analyst', 'programmer', 'system', 'project', 'web', 'devel', 'chief']
    titleLc = row['emp_title'].lower()
    if titleLc.startswith('it ') or titleLc.endswith(' it'):   
        return 1

    if 'sales' in titleLc:
        return 0

    return any([pos in row['emp_title'].lower() for pos in posList])

@FexMethod('EmployeeOmit')
def fex_omit_empl(row):
    return len(row['emp_title'].strip()) < 3

@FexMethod('EmployYears')
def fex_empyears(row):
    val = row['emp_length'].lower()
    if '10+' in val:
        return 10.0
    elif 'n/a' in val:
        return 0
    elif '< 1' in val:
        return 0.5
    else:
        return float(val.split()[0])


otherCols = ['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'sub_grade', 'url']
@OtherDataMethod(*otherCols)
def getOtherCols(row):
    return [row[n].strip() for n in otherCols]


####################################################################################################
####################################################################################################
# SCORING FUNCTION
####################################################################################################
####################################################################################################
# Haven't figured this part out yet...
#def ScoringEvalRules():
    #ScoringRules = \
    #{
        ## Feature                    [isFilter, evalHigh,  ScoringWeight]
        #'HomeOwn_other':             [ +1,  -0.2]  # higher values should subtract score
        #'Purpose_small_business':    [ +1, -0.2,

