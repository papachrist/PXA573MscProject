import os
import numpy as np
import pickle
import random
import math

def readClasses():
    import csv
    classes = []
    with open('/media/papachristo/My Passport/finalp/Classes.csv', mode='r') as infile:
        reader = csv.reader(infile)

        for rows in reader:
            k = rows
            # v = rows[1]
            classes.append(k)
    return classes

def identactions(lt, top):
    acdict = dict()
    actions = []
    cl = readClasses()
    for l in lt[1]:
        if l not in actions:
            actions.append(l)
            acdict[str(l)] = 1
        elif len(actions) != 0:
            acdict[str(l)] = acdict.get(str(l), 0) + 1

    print(acdict)
    tk = np.load(top)

    for whac in actions:
        score = 0
        c = 0
        print(cl[whac])
        for l in lt[1]:
            if l == int(whac):
                if tk[c] == 1:
                    score += 1
            c += 1
        tot = acdict.get(str(whac))

        print(score/tot)
    return actions, acdict


tdirectory = '/media/papachristo/My Passport/finalp/MyQualysis/combdata/topk/xsub'
ldirectory = '/media/papachristo/My Passport/finalp/MyQualysis/combdata/Vlf/'
file = input('Input label file name: ')
tfile = file #input('Input top-k file name: ')

lbl = ldirectory + 'L' + file + '.pkl'
top = tdirectory + tfile + '.npy'


pkl_file = open(lbl, 'rb')
labels = pickle.load(pkl_file)
pkl_file.close()

actions, acdict = identactions(labels, top)

