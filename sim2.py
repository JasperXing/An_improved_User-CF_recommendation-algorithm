import pandas
import pandas as pd
import math


def readRatingData(path=''):
    f = pd.read_table(path,sep=',',names=['UserID','ItemID','Rating'],engine='python')
    f.to_csv('train4.csv',index=False)
    return f


def readClassifyData(path=''):
    f = pd.read_table(path,sep=',',names=['UserID','ItemID','Rating'],engine='python')
    f.to_csv('classifying4.csv',index=False)
    return f


def calcuteSimilar(series1,series2):
    unionLen = len(set(series1) & set(series2))
    if unionLen == 0: return 0.0
    product = len(series1) * len(series2)
    similarity = unionLen / math.sqrt(product)
    return similarity


def calcuteSimilar2(UserID1,UserID2):
    classify = readClassifyData(path='C:/Users/xingershang/Desktop/classifying4.csv')
    series1 = pd.DataFrame({'ItemID':classify[classify['UserID'] == UserID1]['ItemID'],
              'Rating':classify[classify['UserID'] == UserID1]['Rating']})
    series2 = pd.DataFrame({'ItemID':classify[classify['UserID'] == UserID2]['ItemID'],
              'Rating':classify[classify['UserID'] == UserID2]['Rating']})
    sum = 0
    for i in series1['ItemID']:
        f1 = list(series1[series1['ItemID'] == i]['Rating'])
        if len(series2[series2['ItemID'] == i]['Rating']) == 0:
            hit = 0
        else:
            f2 = list(series2[series2['ItemID'] == i]['Rating'])
            hit = f1[0]*f2[0]
        sum += hit
    max1 = max(series1['Rating'])
    max2 = max(series2['Rating'])
    similar2 = sum/(math.sqrt(max1*max2))
    return similar2


def calcuteUser2(csvpath,targetID,K,p):
    frame = readRatingData(csvpath)
    targetUser = frame[frame['UserID'] == targetID]['ItemID']
    otherUsersID = [i for i in set(frame['UserID']) if i != targetID]
    otherUsers = [frame[frame['UserID'] == i]['ItemID'] for i in otherUsersID]
    sim1 = [calcuteSimilar(targetUser,user) for user in otherUsers]
    sim2 = [calcuteSimilar2(targetID,others) for others in otherUsersID]
    similarlist = []
    for i in range(0,len(sim1)):
        similarlist.append(p*sim2[i]+(1-p)*sim1[i])
    similarSeries = pd.Series(similarlist,index=otherUsersID)
    return similarSeries.sort_values()[K:]


def calcuteInterest(frame,similarSeries,targetItemID,K,p):
    similarUserID = similarSeries.index
    similarUsers = [frame[frame['UserID'] == i] for i in similarUserID]
    similarUserValues = similarSeries.values
    UserInstItem = []
    for u in similarUsers:
        if targetItemID in u['ItemID'].values: UserInstItem.append(u[u['ItemID']==targetItemID]['Rating'].values[0])
        else: UserInstItem.append(0)
    interest = sum([similarUserValues[v]*UserInstItem[v]/5 for v in range(len(similarUserValues))])
    return interest


def calcuteItem(csvpath,targetUserID,K,TopN,p):
    frame = readRatingData(csvpath)
    similarSeries = calcuteUser2(csvpath=csvpath,targetID=targetUserID,K=K,p=p) 
    userItemID = set(frame[frame['UserID'] == targetUserID]['ItemID'])
    otherItemID = set(frame[frame['UserID'] != targetUserID]['ItemID'])
    ItemID = list(userItemID ^ otherItemID)
    interestList = [calcuteInterest(frame,similarSeries,book,K=K,p=p) for book in ItemID]
    interestSeries = pd.Series(interestList, index=ItemID)
    return interestSeries.sort_values()[-TopN:]


def calcutelist(csvpath,a,b,K,TopN,p):
    frame =readRatingData(csvpath)
    reclist = []
    allUserID = list(set(frame['UserID']))
    for u in allUserID[a:b]:
        reclist.append(calcuteItem(csvpath,targetUserID=u,K=K,TopN=TopN,p=p))
    return reclist


def PrecisionRecall(test, reclist, TopN):
    hit = 0
    n_recall = 0
    n_precision = 0
    allUserID = list(set(reclist['UserID']))
    for i in allUserID:
        te = test[test['UserID'] == i]['ItemID']
        re = reclist[reclist['UserID'] == i]['ItemID']
        hit += len(set(te) & set(re))
        n_recall += len(te)
        n_precision += TopN
    return [hit / (1.0 * n_recall), hit / (1.0 * n_precision)]
