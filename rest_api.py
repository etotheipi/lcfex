#! /usr/bin/python
import os
import json
import requests
import time
from pprint import pprint

MAP_NOTE_VAR_TO_HISTORICAL_DATA = \
{
    # ''    means pass through
    # '___' means to split by uppercase, join with _, lowercase
    # None  means don't transfer
    # Anything else is a direct mapping of note vars to hist vars

    "id":                               '',
    "memberId":                         '',
    "loanAmount":                       'loan_amnt',
    "fundedAmount":                     'funded_amnt',
    "term":                             '',
    "intRate":                          '___',
    "expDefaultRate":                   '',
    "serviceFeeRate":                   '',
    "installment":                      '',
    "grade":                            '',
    "subGrade":                         '___',
    "empLength":                        '___',
    "homeOwnership":                    '___',
    "annualInc":                        '___',
    "isIncV":                           'verification_status',
    "acceptD":                          None,
    "expD":                             None,
    "listD":                            None,
    "creditPullD":                      'last_credit_pull_d',
    "reviewStatusD":                    None,
    "reviewStatus":                     None,
    "desc":                             '',
    "purpose":                          '',
    "addrZip":                          'zip_code',
    "addrState":                        '___',
    "investorCount":                    None,
    "ilsExpD":                          '___',
    "initialListStatus":                '___',
    "empTitle":                         '___',
    "accNowDelinq":                     '___',
    "accOpenPast24Mths":                'acc_open_past_24mths',
    "bcOpenToBuy":                      '___',
    "percentBcGt75":                    'percent_bc_gt_75',
    "bcUtil":                           '___',
    "dti":                              '',
    "delinq2Yrs":                       'delinq_2yrs',
    "delinqAmnt":                       '___',
    "earliestCrLine":                   '___',
    "ficoRangeLow":                     '___',
    "ficoRangeHigh":                    '___',
    "inqLast6Mths":                     'inq_last_6mths',
    "mthsSinceLastDelinq":              '___',
    "mthsSinceLastRecord":              '___',
    "mthsSinceRecentInq":               '___',
    "mthsSinceRecentRevolDelinq":       '___',
    "mthsSinceRecentBc":                '___',
    "mortAcc":                          '',
    "openAcc":                          '___',
    "pubRec":                           '___',
    "totalBalExMort":                   '___',
    "revolBal":                         '___',
    "revolUtil":                        '___',
    "totalBcLimit":                     '___',
    "totalAcc":                         '___',
    "totalIlHighCreditLimit":           '___',
    "numRevAccts":                      '___',
    "mthsSinceRecentBcDlq":             '___',
    "pubRecBankruptcies":               '___',
    "numAcctsEver120Ppd":               'num_accts_ever_120_pd',
    "chargeoffWithin12Mths":            'chargeoff_within_12_mths',
    "collections12MthsExMed":           'collections_12_mths_ex_med',
    "taxLiens":                         '___',
    "mthsSinceLastMajorDerog":          '___',
    "numSats":                          '___',
    "numTlOpPast12m":                   '___',
    "moSinRcntTl":                      '___',
    "totHiCredLim":                     '___',
    "totCurBal":                        '___',
    "avgCurBal":                        '___',
    "numBcTl":                          '___',
    "numActvBcTl":                      '___',
    "numBcSats":                        '___',
    "pctTlNvrDlq":                      '___',
    "numTl90gDpd24m":                   'num_tl_90g_dpd_24m',
    "numTl30dpd":                       'num_tl_30dpd',
    "numTl120dpd2m":                    'num_tl_120dpd_2m',
    "numIlTl":                          '___',
    "moSinOldIlAcct":                   None,
    "numActvRevTl":                     '___',
    "moSinOldRevTlOp":                  '___',
    "moSinRcntRevTlOp":                 '___',
    "totalRevHiLim":                    '___',
    "numRevTlBalGt0":                   'num_rev_tl_bal_gt_0',
    "numOpRevTl":                       '___',
    "totCollAmt":                       '___',
    "applicationType":                  '___',
    "annualIncJoint":                   '___',
    "dtiJoint":                         '___',
    "isIncVJoint":                      'verification_status_joint',
    "openAcc6m":                        'open_acc_6m',
    "openActIl":                        None,
    "openIl12m":                        'open_il_12m',
    "openIl24m":                        'open_il_24m',
    "mthsSinceRcntIl":                  '___',
    "totalBalIl":                       '___',
    "iLUtil":                           'il_util',
    #"ilUtil":                           '___',
    "openRv12m":                        'open_rv_12m',
    "openRv24m":                        'open_rv_24m',
    "maxBalBc":                         '___',
    "allUtil":                          '___',
    "inqFi":                            '___',
    "totalCuTl":                        '___',
    "inqLast12m":                       'inq_last_12m',
    "mtgPayment":                       None,
    "housingPayment":                   None,
    "revolBalJoint":                    None,
    #"secAppFicoRangeLow":               '___',
    #"secAppFicoRangeHigh":              '___',
    #"secAppEarliestCrLine":             '___',
    #"secAppInqLast6Mths":               'sec_app_inq_last_6mths',
    #"secAppMortAcc":                    '___',
    #"secAppOpenAcc":                    '___',
    #"secAppRevolUtil":                  '___',
    #"secAppOpenActIl":                  '___',
    #"secAppNumRevAccts":                '___',
    #"secAppChargeoffWithin12Mths":      'sec_app_chargeoff_within_12_mths',
    #"secAppCollections12MthsExMed":     'sec_app_collections_12_mths_ex_med',
    #"secAppMthsSinceLastMajorDerog":    'sec_app_mths_since_last_major_derog',
    #"disbursementMethod":               None,
}


def ConvertNote2HistVariables(noteMap):
    outMap = {}
    for noteName,histName in MAP_NOTE_VAR_TO_HISTORICAL_DATA.iteritems():
        if histName is None:
            continue

        noteVar = noteMap.get(noteName, [None])
        if noteVar is None:
            noteVar = ''

        try:
            noteVar = str(noteVar)
        except:
            # If the above fails, it's because of unicode problems
            noteVar = str(unicodedata.normalize('NFKD', noteVar).encode('ascii','ignore'))

        if noteVar == [None]:
            print 'Warning: note does not have var: %s' % noteName
            noteVar = ''

        elif len(histName) == 0:
            outMap[noteName] = noteVar

        elif histName == '___':
            newHistName = ''
            for char in noteName:
                if ord('A') <= ord(char) <= ord('Z'):
                    newHistName += '_'+char.lower()
                else:
                    newHistName += char
            outMap[newHistName] = noteVar
        else:
            outMap[histName] = noteVar

       

    # Tweaks to certain vars we know we need
    outMap['verification_status'] = outMap['verification_status'].replace('_',' ')
    outMap['term'] = ' %s months' % outMap['term']
    outMap['title'] = ''
    outMap['desc'] = '' if outMap['desc'] is None else outMap['desc']
    outMap['revol_util'] += '%'
    outMap['total_pymnt'] = '0'
    outMap['url'] = 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=%s' % outMap['id']
    
    # emp_length should be a string (hist data uses "10+ years", "n/a")
    if len(outMap['emp_length']) == 0:
        outMap['emp_length'] = '0'
    else:
        outMap['emp_length'] = str(int(int(outMap['emp_length'])/12.0 + 0.5))

    return outMap
            


def LcApiCall(action, resource, subresource, urlExtra=None, postPayload=None):
    """
    If resource is "account", then accountId will be used as the next URL path segment

    Example resource/subresource pairs
      - accounts/summary
      - accounts/availablecash
      etc
      - loans/listing
    """
    credFile = json.load(open(os.path.expanduser('~/.lcapi'), 'r'))
    headers = {}
    headers['Accept'] = 'application/json'
    headers['Authorization'] = credFile['apiKey']

    if action.lower()=='post':
        headers['Content-type'] = credFile['application/json']

    if resource=='accounts':
        resource = 'accounts/' + credFile['investorId']
    
    URI = 'https://api.lendingclub.com/api/investor/v1/%s/%s' % (resource, subresource)

    if urlExtra is not None:
        URI += '?' + '&'.join(['%s=%s'%(k,v) for k,v in urlExtra.iteritems()])

    print 'Executing action=%s via URL: %s' % (action, URI)
    if action.lower()=='get':
        return requests.get(URI, headers=headers).json()
    elif action.lower()=='post':
        return requests.post(URI, headers=headers, data=postPayload).json()

def GetLoanListing(getAllAvail=False):
    urlExtra = {'showAll':'true'} if getAllAvail else None
    loanList = LcApiCall('GET', 'loans', 'listing', urlExtra=urlExtra)['loans']
    return [ConvertNote2HistVariables(note) for note in loanList]
    
def GetAlreadyInvestedIds():
    notes = LcApiCall('get', 'accounts', 'notes')['myNotes']
    return set([str(n['loanId']) for n in notes])

def GetPortfolioIdByName(pname):
    portfolios = LcApiCall('get', 'accounts', 'portfolios')['myPortfolios']
    for port in portfolios:
        if port['portfolioName'] == pname:
            return port['portfolioId']
    else:
        print 'Portfolio with name "%s" does not exist' % pname
        return None

def GetNewLoanIdsAboveThresh(scoreCols, scoreMtrx, thresh):
    idCol     = scoreCols.index('id')
    dupCol    = scoreCols.index('Dup')
    scoreCol  = scoreCols.index('SCORE')

    idList = []
    for scoreRow in scoreMtrx:
        # The dup column is either '' or '*'
        if len(scoreRow[dupCol].strip()) > 0:
            continue 

        if float(scoreRow[scoreCol]) >= thresh:
            idList.append(scoreRow[idCol])

    return idList


def CreateInvestOrderPayload(loanIdAmtPairs, portfolioNameOrId=None, outFile=None):

    if isinstance(portfolioNameOrId, basestring):
        if not portfolioNameOrId.isdigit():
            portfolioNameOrId = GetPortfolioIdByName
    
    portfolioId = portfolioNameOrId
    investorId = json.load(open(os.path.expanduser('~/.lcapi'), 'r'))['investorId']
    alreadyInvested = GetAlreadyInvestedIds()
        
    orderStruct = {'aid': investorId,
                   'orders': [] }

    for loanId,amt in loanIdAmtPairs:
        noteOrder = { 'loanId':            loanId,
                      'requestedAmount':   int(amt),
                      'portfolioId':       portfolioId}

        if loanId in alreadyInvested:
            print 'WARNING: Your account already has an investment in loan: %d' % loanId
        print 'https://www.lendingclub.com/browse/addToPortfolio.action?loan_id=%s&loan_amount=%d' % (loanId, int(amt))

        orderStruct['orders'].append(noteOrder)
        
    print 'Payload for order'
    print json.dumps(orderStruct, indent=2)

    payloadFile = outFile
    if payloadFile is None:
        payloadFile = 'proposed_order_payload_%s.json' % time.strftime('%Y-%m-%d-%H%M')

    with open(payloadFile, 'w') as f:
        f.write(json.dumps(orderStruct, indent=2))
    print 'Wrote proposed order to:', payloadFile

    return orderStruct


def ExecuteInvestmentFromPayloadFile(payloadFile):
    payload = json.loads(open(payloadFile,'r').read())
    response = LcApiCall('post', 'accounts', 'orders', postPayload=payload)
    print json.dumps(response, indent=2)




#loanList = LcApiCall('GET', 'loans', 'listing')['loans']
#loanList = LcApiCall('GET', 'loans', 'listing', urlExtra={'showAll':'true'})
#pprint(loanList[0])
#pprint(ConvertNote2HistVariables(loanList[0]))
#pprint(LcApiCall('get', 'accounts', 'notes'))
#pprint(LcApiCall('get', 'accounts', 'portfolios'))
#print 'Porfolio ID:', GetPortfolioIdByName('Initial_Stats_1119')
