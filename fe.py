#! /usr/bin/python
import csv
import math
from datetime import datetime
from dateutil import parser as dateParser
import os
import hashlib
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
           ]



####################################################################################################
#    A singleton class that tracks all the feature extraction methods and applies them
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
        self.numRows = 0
        self.numFeatures = 0
        self.avgVec = []
        self.stdVec = []
        self.binaryList = []

    def RunFeatureExtractor_Single(self, row, context):

        if context.lower() == 'train':
            if not row['loan_status'].lower() in ['fully paid', 'charged off']:
                return

        # This checks all the filters in order, skipping this row if any fail
        #if context=='predict':
            #print '-'*80
            #for k,v in row.iteritems():
                #print '  ', k.ljust(40), v

        #print ''
        for fn in FeatureExtractor.FilterList:
            #print fn.__name__,
            if not fn(row):
                return

        # Run all the fex functions 
        nList,vList = [],[]
        nList.append('_CLASS_')
        if context=='train':
            vList.append(ClassificationFunc(row))
        else:
            vList.append(-1)

        for func,names in FeatureExtractor.FeatureList:
            nList.extend(names)
            vList.extend([float(v) for v in func(row)])


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


    def RunFeatureExtractor(self, rowIterator, context):
        # This returns the column headers and matrix as a numpy object for efficient calcs
        # It also returns column headers and a list-of-lists that has all the same plus extra cols
        self.fexMatrix = None

        for i,row in enumerate(rowIterator):
            self.RunFeatureExtractor_Single(row, context=context)

        # This can return an error if self.fexMatrix skipped all rowx
        if self.fexMatrix is None:
            print 'Error running fex-matrix:  all rows failed filters'
            exit(1)
            
        self.fexMatrix.resize(self.numRows, self.numFeatures)
        self.allDataMatrix = self.allDataMatrix[:self.numRows]

        self.avgVec = np.mean(self.fexMatrix, axis=0)
        self.stdVec = np.std(self.fexMatrix, axis=0)

        # This simply sets binaryVec[c] to False if any row has a value that isn't 0 or 1
        self.binaryList = [True] * self.numFeatures
        for row in self.fexMatrix:
            for c in range(self.numFeatures):
                if row[c] not in [0,1]:
                    self.binaryList[c] = False
        

    def GetFexMatrix(self):
        return self.colNames, self.fexMatrix
 

    def GetAllDataListOfLists(self):
        return self.allDataColNames, self.allDataMatrix


    def GetColumnStats(self):
        return self.avgVec, self.stdVec, self.binaryList

    
    def RunBinaryStats(self):
        """
        This is for exploring data relationships in historical data.  
        Correlation coefficients are pretty worthless for columns that are binary flags,
        especially so the further it is from 50/50 trues and falses.  Instead, look
        at result ratios for the subset of each column which has this flag true
        """
        allClassSum = 0
        featureRowCount = [0]*self.numFeatures
        featureClassSum = [0]*self.numFeatures
        for row in self.fexMatrix:
            allClassSum += row[0]
            for c in range(self.numFeatures):
                if self.binaryList[c]:
                    if row[c]==1:
                        featureClassSum[c] += row[0]
                        featureRowCount[c] += 1
                
        ratioList = [0] * self.numFeatures
        for c in range(self.numFeatures):
            if featureRowCount[c] > 0:
                ratioList[c] = float(featureClassSum[c]) / float(featureRowCount[c])
        
        totalAvg = float(allClassSum) / float(self.numRows)
        return ratioList, featureClassSum, featureRowCount, totalAvg
            

    def ScoreLoans(self, allDataMatrix=None, 
                         gradeMin='A1', 
                         gradeMax='G5', 
                         outFile='scoredLoans.csv', 
                         weightByRate=None,
                         doRandom=False,
                         alreadySet=None):

        if alreadySet is None:
            alreadySet = set([])

        if allDataMatrix is None:
            allDataMatrix = self.allDataMatrix

        # Create a new matrix with normalized data
        normalizedMtrx = [ [0]*len(self.allDataColNames) for i in range(len(allDataMatrix)) ]

        for r,row in enumerate(allDataMatrix):
            for c in range(len(self.allDataColNames)):
                if c >= self.numFeatures or self.stdVec[c]==0 or self.binaryList[c]:
                    normalizedMtrx[r][c] = row[c]
                else:
                    normalizedMtrx[r][c] = (float(row[c]) - self.avgVec[c]) / self.stdVec[c]
    
        srules = FeatureExtractor.GetScoringRules()
        scoreRowWgtList = []
        sortedRulesKeys = sorted(srules.keys())
        colIndices = [self.allDataColNames.index(k) for k in sortedRulesKeys]


        gradeCol = self.allDataColNames.index('GradeFloat')
        irateCol = self.allDataColNames.index('IntRate')
        approxIntCol = self.allDataColNames.index('ApproxIntRate')
        loanIdCol = self.allDataColNames.index('id')

        for r,row in enumerate(normalizedMtrx):
            rowGradeVal = allDataMatrix[r][gradeCol]
            if not (GRADE2VALUE(gradeMin) <= rowGradeVal <= GRADE2VALUE(gradeMax)):
                continue

            score = 0
            weights = []
            #print ''
            if doRandom:
                loadId = allDataMatrix[r][loanIdCol]
                last3 = [ord(c) for c in hashlib.sha256(str(loadId)).digest()[-3:]]
                scoreRowWgtList.append( (last3[0]*256*256 + last3[1]*256 + last3[0], r, [0]) )
                continue


            for c,col in enumerate(sortedRulesKeys):
                #print c,col,colIndices[c],row[colIndices[c]],'|'
                func = srules[col]
                scoreMod = func(row[colIndices[c]])
                if scoreMod is None:
                    break

                # Passing a Requires method returns 0 so does not affect score
                score += scoreMod
                weights.append(scoreMod)
                
            else:
                # This runs if the row didn't fail any requirements, row and score should be added
                if score > 0:
                    if weightByRate is not None:
                        # Use a non-None float value, usually 0.5 or 1.0 for weightByRate 
                        score = score * (weightByRate + allDataMatrix[r][irateCol])
                    scoreRowWgtList.append( (score, r, weights) )
                 
            
        print '-'*80
        if doRandom:
            print 'Performance over randomized set'
        else:
            print 'Performance over data set'

        with open(outFile, 'w') as f:
            cols = ['SCORE', 'CumulAvgInterest','Dup']
            cols.extend(self.allDataColNames)
            cols.extend(['WEIGHT_' + k for k in sortedRulesKeys])
            
            
            f.write(','.join(cols) + '\n')
            outMatrix = []
            rowCount,sumEffInt = 0,0
            for score,rowNum,wgts in sorted(scoreRowWgtList, reverse=True):
                rowCount += 1
                sumEffInt += allDataMatrix[rowNum][approxIntCol]
                intStr = '%0.3f' % (sumEffInt/float(rowCount))
                dupStr = '*' if str(allDataMatrix[rowNum][loanIdCol]) in alreadySet else ''
            
                if rowCount > 0 and rowCount%100==0:
                    print 'Cumulative ROI for first %d loans: %s (score thresh: %0.1f)' % \
                                                                     (rowCount, intStr, score)
                outMatrix.append([score, intStr, dupStr] + allDataMatrix[rowNum][:] + wgts)
                f.write(','.join([str(v) for v in outMatrix[-1][:]]) + '\n')
                if len(outMatrix) > 501:
                    break

        return outMatrix
                
        

    ###############################################################################################
    ###############################################################################################
    # SCORING FUNCTION
    ###############################################################################################
    ###############################################################################################
    @staticmethod
    def GetScoringRules():
    
        def normalizeStdev(stdev, high=2.5, med=1.75, low=1.0):
            if   stdev >  high: return  1.0
            elif stdev >  med:  return  0.6
            elif stdev >  low:  return  0.3
            elif stdev > -low:  return  0.0
            elif stdev > -med:  return -0.3
            elif stdev > -high: return -0.6
            else:               return -1.0
            
        HigherIsBetter  = lambda weight: (lambda val: normalizeStdev(val) * weight)
        LowerIsBetter   = lambda weight: (lambda val: normalizeStdev(val) * weight * -1)
        PreferTrue      = lambda weight: (lambda val:  weight if val==1 else 0)
        PreferFalse     = lambda weight: (lambda val: -weight if val==1 else 0)
        
        # Calling method should omit if none
        RequireTrue     = lambda: (lambda val: None if not val else 0)
        RequireFalse    = lambda: (lambda val: None if val else 0)
        RequirePosStdev = lambda stdev: (lambda val: None if val < stdev else 0)
        RequireNegStdev = lambda stdev: (lambda val: None if val > -stdev else 0)
        
        ScoringRules = \
        {
            # These are filtering rules
            'Purpose_small_business':    RequireFalse(),
            'TitleCapNoneOrAll':         RequireFalse(),
            'EmpTitleCapNoneOrAll':      RequireFalse(),
            'EmployeeOmit':              RequireFalse(),
            'HomeOwn_other':             RequireFalse(),

            # These are weight adjustment rules
            'EmployeeTech':              PreferTrue(     weight=10 ),
            'Purpose_credit_card':       PreferTrue(     weight=10 ),
            'LogMonthlyIncome':          HigherIsBetter( weight=10 ),
            'Purpose_major_purchase':    PreferTrue(     weight=6  ),
            'MonthlyIncome2Pmt':         HigherIsBetter( weight=6  ),
            'dti':                       LowerIsBetter(  weight=6  ),
            'inq_last_6mths':            LowerIsBetter(  weight=6  ),
            'HomeOwn_mortgage':          PreferTrue(     weight=5  ),
            'EmployeeLeader':            PreferTrue(     weight=3  ),
            'OldestCreditYrs':           HigherIsBetter( weight=3  ),
            'LastDelinqAgo':             LowerIsBetter(  weight=3  ),

            # The weighting is lower for each of these, knowing they will stack (for instance,
            # if numAccts is 1 it will get a pos score <2, <3, etc.  Somehow, our tet data
            # indicates almost exactly .6% default rate reduced for every acct<6
            'NumAccounts_lt_1':          PreferTrue(  weight=1  ),
            'NumAccounts_lt_2':          PreferTrue(  weight=2  ),
            'NumAccounts_lt_3':          PreferTrue(  weight=2  ),
            'NumAccounts_lt_4':          PreferTrue(  weight=2  ),
            'NumAccounts_lt_5':          PreferTrue(  weight=1  ),
            'NumAccounts_lt_6':          PreferTrue(  weight=1  ),

            # Smae store for num inquiries.  Almost 0.7% reduced rate below 2 
            'NumInquiries_lt_1':         PreferTrue(  weight=1.5  ),
            'NumInquiries_lt_2':         PreferTrue(  weight=1.5  ),
        }

        return ScoringRules



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

# These are columns that will be read and converted to floats directly into the feature matrix
passthruFields = \
[
    'dti', 
    'open_acc', 
    'total_acc', 
    'delinq_2yrs', 
    'acc_open_past_24mths', 
    'bc_util', 
    'num_actv_bc_tl', 
    'inq_last_6mths', 
    'pub_rec', 
    'num_sats', 
    'percent_bc_gt_75',
]

####################################################################################################
# Filtering Methods
####################################################################################################
@FilterMethod
def filter_term36_only(row):
    return row['term'].strip().startswith('36') 

@FilterMethod
def filter_grade_b2d_only(row):
    return GRADE2VALUE('B1') <= GRADE2VALUE(row['sub_grade']) < GRADE2VALUE('F')

@FilterMethod
def filter_annual_inc_gt_40k(row):
    return float(row['annual_inc']) >= 40000

@FilterMethod
def filter_nonempty_vars(row):
    for i,c in enumerate(passthruFields):
        if len(str(row[c]).strip()) == 0:
            return False
    else:
        return True

@FilterMethod
def filter_has_revolving_util_value(row):
    return '%' in row['revol_util']

@FilterMethod
def filter_apptype(row):
    return row['application_type'].lower() == 'individual'

####################################################################################################
# Feature Extraction Methods
####################################################################################################
@FexMethod('GradeFloat')
def fex_lc_grade_float(row):
    # A: [0,1), B: [1,2), C: [2,3), D: [3,4), E: [4,5), F: [5,6)
    return GRADE2VALUE(row['sub_grade'])

@FexMethod('IntRate')
def fex_intrate(row):
    if isinstance(row['int_rate'], float):
        return row['int_rate']
    else:
        return float(row['int_rate'].rstrip('%'))/100.0

# This is all fields in the spreadsheet that we want to analyze and require no manipulation
@FexMethod(*passthruFields)
def fex_allpassthru(row):
    return [float(row[n]) for n in passthruFields]

@FexMethod('MonthlyIncome2Pmt')
def fex_install_per_income(row):
    moPayment = float(row['installment'])
    moIncome  = float(row['annual_inc'])/12
    return min(moIncome/moPayment, 50)

@FexMethod('LogMonthlyIncome')
def fex_log_income(row):
    return math.log(float(row['annual_inc']))

@FexMethod(*['HomeOwn_'+t for t in HOME_OWNERSHIPS])
def fex_home(row):
    return BitListFromSet(row['home_ownership'], HOME_OWNERSHIPS)

@FexMethod(*['Purpose_'+t for t in PURPOSES])
def fexmulti_purpose(row):
    return BitListFromSet(row['purpose'], PURPOSES)

@FexMethod('OldestCreditYrs')
def fex_oldest_credit_line(row):
    dtOrig = dateParser.parse(row['earliest_cr_line']).replace(tzinfo=None)
    dtNow  = datetime.now().replace(tzinfo=None)
    return min((dtNow - dtOrig).days / 365.0, 20)

@FexMethod('VerifyStatus')
def fex_verify_stat(row):
    vstr = row['verification_status'].lower()
    if vstr=='not verified':
        return 0
    elif vstr=='verified':
        return 0.5
    elif vstr=='source verified':
        return 1.0
    else:
        raise Exception('Unknown verify status: %s' % vstr)

@FexMethod('RevolvUtil')
def fex_revolveutil(row):
    if isinstance(row['revol_util'], float):
        return row['revol_util']
    else:
        return float(row['revol_util'].rstrip('%')) / 100.

@FexMethod('Revolv2Income')
def fex_openaccts(row):
    revBal = float(row['revol_bal'])
    annualInc = float(row['annual_inc'])
    return revBal/annualInc

@FexMethod('TitleCapNoneOrAll')
def fex_titlecapnoneallcaps(row):
    # All caps or no caps
    t = row['title'].strip()
    return (t in [t.lower(), t.upper()]) and len(t)>0

@FexMethod('EmpTitleCapNoneOrAll')
def fex_titlecapnoneallcaps(row):
    # All caps or no caps
    t = row['emp_title'].strip()
    return t in [t.lower(), t.upper()] and len(t)>0

@FexMethod('HasDescr')
def fex_buyerdescription_nonzero(row):
    return len(row['desc'].strip()) == 0 or row['desc']==row['desc'].lower()

@FexMethod('DescrLength')
def fex_buyerdescription_nonzero(row):
    desc = row['desc'].strip()
    desc = desc.replace('Borrower added on','')
    desc = desc.replace('<br>', '')
    if len(desc) > 0 and ' > ' in desc:
        pcs = desc.split(' > ')
        if len(pcs) == 2:
            desc = pcs[1]
        else:
            desc = pcs[1][:-8]
        
    # Kind of a logarithmic func...
    l = len(desc)
    if l == 0:    return 0
    elif l < 30:  return 1
    elif l < 50:  return 2
    elif l < 80:  return 4
    elif l < 120: return 5
    elif l < 200: return 6
    elif l < 400: return 7
    else:         return 8
    return len(desc)

@FexMethod('EmployeeTech', 'EmployeeLeader', 'EmployeeMedical')
def fex_employee_keywords(row):
    """
    This is a little weird because I don't want someone who has multiple keywords to get too much
    credit for these features.  So I sort by the ones that are most important, only return one
    """
    out = [False, False, False]
    
    techList = ['engineer', 'tech', 'info', 'analyst', 'programmer', 'system', 'project', 'web', 'devel', 'chief']
    leadList = ['supervisor','manager','vp', 'admin', 'senior', 'officer', 'lead', 'director', 'president']
    medList = ['nurse', 'drug', 'doctor', 'resident', 'phys', 'medical', 'medicine', 'hospital', 'ologist', 'ometrist', 'registered', ' rn ']

    titleLc = row['emp_title'].lower()
    if titleLc.startswith('it ') or titleLc.endswith(' it'):   
        out[0] = True
    elif any([pos in titleLc for pos in techList]):
        out[0] = True
    elif any([pos in titleLc for pos in leadList]):
        out[1] = True
    elif titleLc.startswith('rn ') or titleLc.endswith(' rn'):   
        out[2] = True
    elif any([pos in titleLc for pos in medList]):
        out[2] = True

    return out

@FexMethod('EmployeeOmit')
def fex_omit_empl(row):
    return len(row['emp_title'].strip()) < 2

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

@FexMethod('LastDelinqAgo')
def fex_lastDelinq(row):
    val = row['mths_since_last_delinq'].strip()
    if len(val)==0 or float(val) > 60:
        return 60.0
    else:
        return float(val)

@FexMethod('LastMajorDerog')
def fex_lastDerog(row):
    val = row['mths_since_last_major_derog'].strip()
    if len(val)==0 or float(val) > 60:
        return 60.0
    else:
        return float(val)

@FexMethod(*['NumDelinq_lt_%d' % i for i in range(1,6)])
def fex_numdelinq_binary(row):
    return [(int(row['delinq_2yrs']) < i) for i in range(1,6)]

@FexMethod(*['NumInquiries_lt_%d' % i for i in range(1,8)])
def fex_numinq_binary(row):
    return [(int(row['inq_last_6mths']) < i) for i in range(1,8)]

@FexMethod(*['NumAccounts_lt_%d' % i for i in range(1,8)])
def fex_numaccts_binary(row):
    return [(int(row['acc_open_past_24mths']) < i) for i in range(1,8)]


####################################################################################################
# These are vars that we need available in final output, but not part of analysis
otherCols = ['id', 'loan_amnt', 'funded_amnt', 'term', 'total_pymnt', 'int_rate', 'sub_grade', 'url']
@OtherDataMethod(*otherCols)
def getOtherCols(row):
    return [row[n].strip() for n in otherCols]

@OtherDataMethod('ApproxIntRate')
def getApproxFinalIntRate(row):
    # Conversion from totalInterest/principal to annualized net return is 1.6-1.7
    return (float(row['total_pymnt']) / float(row['funded_amnt']) - 1) / 1.65

