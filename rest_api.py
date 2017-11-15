#! /usr/bin/python
import os
import json
import requests
from pprint import pprint

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


#pprint(LcApiCall('GET', 'loans', 'listing', urlExtra={'showAll':'true'}))
pprint(LcApiCall('GET', 'loans', 'listing'))
