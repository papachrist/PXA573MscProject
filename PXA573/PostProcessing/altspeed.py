import os
import numpy as np
import pickle
import random
import math

def altspeed(f, frate):
    newdata = []

    # LOAD DATA AND LABELS
    data = directory + f + '.npy'
    actions = np.load(data)
    print(actions.shape)
    for a in actions:
        action = []
        for cr in a:
            crn = []
            for ind in range(len(cr)):
                    if (ind % frate) == 0 and len(crn) < 300:
                        fr = cr[ind]
                        crn.append(fr)
            action.append(crn)
        newdata.append(action)

    print(np.asarray(newdata).shape)
    # output them
    return newdata

def save(dt):

    outfile = '/media/papachristo/My Passport/finalp/MyQualysis/combdata/25Hz/'

    # dv, lv = combval()
    name = input('File name:')
    #
    # # TRAIN
    np.save(outfile + name + '.npy', np.asarray(np.float32(dt)))



directory = '/media/papachristo/My Passport/finalp/MyQualysis/combdata/100Hz/'
file = input('Input file name: ')

dt = altspeed(file, 5)

sve = input('Save files [y/n]? ')
if sve == 'y':
    save(dt)
