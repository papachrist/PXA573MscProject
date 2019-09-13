import os
import numpy as np
import pickle
import random
import math

rightarm = [[24,12], [25,12], [12,11], [11,10]]
leftarm  = [[6,7], [7,8], [8,22], [8,23]]
ucentrbody = [[21,3], [4,3], [5,6], [9,10]]

lcentrbody = [[9,21], [21, 5], [21,2], [2,1]] # [1,17], [1,13]]
leftleg  = [[13,14],[14,15], [15,16], [1, 17]]
rightleg = [[17,18],[18,19], [19,20], [1, 13]]

upperbones = []
lowerbones = []

upperbones.append(rightarm)
upperbones.append(leftarm)
upperbones.append(ucentrbody)
lowerbones.append(lcentrbody)
lowerbones.append(leftleg)
lowerbones.append(rightleg)

bones = []
bones.extend(rightarm)
bones.extend(leftarm)
bones.extend(ucentrbody)
bones.extend(lcentrbody)
bones.extend(leftleg)
bones.extend(rightleg)


upperbones = np.array(upperbones) - 1
lowerbones = np.array(upperbones) - 1
bones = np.array(bones) - 1


def whichbones():

    selbody = []
    otherbody = []

    occb = []

    # newdata = []
    # data = directory + f + '.npy'
    # actions = np.load(data)
    # print(actions.shape)

    N = math.ceil(random.uniform(2, 4))
    S = math.ceil(random.uniform(0, 2))

    if S == 1:
        selbody = upperbones
        otherbody = lowerbones
    else:
        selbody = lowerbones
        otherbody = upperbones

    H = math.ceil(random.uniform(0, N))
    R = N - H

    occbones = 0
    c = 0
    while(len(occb) < N-1):
        if len(occb) < H:
            rg = math.ceil(random.uniform(0, 2))

            G = math.ceil(random.uniform(0, H))
            
            for i in range(0, G):
                    ran = math.ceil(random.uniform(0, 3))
                    if len(occb) < H:
                        occb.append([selbody[rg][ran]])
        else:
            rg = math.ceil(random.uniform(0, 2))

            G = math.ceil(random.uniform(0, R))


            for i in range(0, G):
                ran = math.ceil(random.uniform(0, 3))
                if len(occb) < N:
                    occb.append([otherbody[rg][ran]])

    print(len(occb))
    print(occb)
    return occb

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

        b = whichbones()
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


directory = '/media/papachristo/My Passport/finalp/MyQualysis/combdata/Vlf/'
file = input('Input file name: ')

cmb = input('Do you want to distort the data in a combination of occlusion and noise [y/n]? ')
if cmb == 'y':
    dt = removebones(file)
    dn = addnoise(file, dt)
    sve = input('Save files [y/n]? ')
    if sve == 'y':
        save(dn)

rmb = input('Do you want to occlude bones [y/n]? ')
if rmb == 'y':
    dt = removebones(file)
    sve = input('Save removed bones files [y/n]? ')
    if sve == 'y':
        save(dt)


adn = input('Do you want to add noise to the data [y/n]? ')
if adn == 'y':
    dn = addnoise(file, dt)

    sve = input('Save noise files [y/n]? ')
    if sve == 'y':
        save(dn)
