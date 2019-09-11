import os
import numpy as np
import pickle
import random
import math

rightarm = [[24,12], [25,12], [12,11], [11,10], [9,10]]
leftarm  = [[21,5], [5,6], [6,7], [7,8], [8,22], [8,23]]
mainbody = [[9,21], [21,3], [4,3], [21,2], [2,1], [1,17], [1,13]]
leftleg  = [[13,14],[14,15], [15,16]]
rightleg = [[17,18],[18,19], [19,20]]
bones = []
bones.extend(rightarm)
bones.extend(leftarm)
bones.extend(mainbody)
bones.extend(leftleg)
bones.extend(rightleg)

bones = np.array(bones) - 1


def removebones(f):

    newdata = []

    # LOAD DATA AND LABELS
    data = directory + f + '.npy'
    actions = np.load(data)
    print(actions.shape)
    for a in actions:
        action = []
        b = []
        for i in range(3):
            b.extend(bones[math.ceil(random.uniform(0, len(bones)-1))])
        for cr in a:
            crn = []
            for fr in cr:
                frn = []
                for ind in range(len(fr)):
                    if ind in b:
                        j = [0, 0]
                    else:
                        j = fr[ind]
                    frn.append(j)
                crn.append(frn)
            action.append(crn)
        newdata.append(action)

    print(np.asarray(newdata).shape)
    # output them
    return newdata


def addnoise(f, dt):

    newdata = []

    # LOAD DATA AND LABELS
    data = directory + f + '.npy'
    actions = np.asarray(np.float32(dt)) #np.load(data)
    print(actions.shape)
    for a in actions:
        action = []
        for cr in a:
            crn = []
            for fr in cr:
                frn = []
                for ind in range(len(fr)):
                    # ms = fr[1]
                    j = fr[ind]
                    perc = random.uniform(0.9, 1.1)
                    fj = perc*j
                    frn.append(fj)
                crn.append(frn)
            action.append(crn)
        newdata.append(action)

    print(np.asarray(newdata).shape)
    # output them
    return newdata

def save(dt):

    outfile = '/media/papachristo/My Passport/finalp/MyQualysis/combdata/newtr/'

    # dv, lv = combval()
    name = input('File name:')
    #
    # # TRAIN
    np.save(outfile + name + '.npy', np.asarray(np.float32(dt)))


directory = '/media/papachristo/My Passport/finalp/MyQualysis/combdata/'
file = input('Input file name: ')

dt = removebones(file)

# sve = input('Save removedbones files [y/n]? ')
# if sve == 'y':
#     save(dt)

dn = addnoise(file, dt)

sve = input('Save noise files [y/n]? ')
if sve == 'y':
    save(dn)
