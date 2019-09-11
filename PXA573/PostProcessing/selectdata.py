import os
import numpy as np
import pickle
import random
import math


def selectop(f):
    combdata = []
    comblabl = []
    combtitl = []
    loadata = []
    labl = []

    # LOAD TOP VALUES
    # tkf = input('Topk file: ')

    topf = '/media/papachristo/My Passport/finalp/MyQualysis/combdata/topkcsvl.npy'
    topk = np.load(topf)
    k = int(input('Input top value: '))
    # LOAD DATA AND LABELS
    data = directory + f + '.npy'
    Labels = directory + 'L' + f + '.pkl'

    pkl_file = open(Labels, 'rb')
    labels = pickle.load(pkl_file)
    pkl_file.close()

    valdata = np.load(data)
    print(valdata.shape)
    for ind in range(len(valdata)):
        if topk[ind] == k:
            combdata.append(valdata[ind])
            comblabl.append(int(labels[1][ind]))
            combtitl.append(labels[0][ind])

    labl.append(combtitl)
    labl.append(comblabl)

    print(np.asarray(combdata).shape)
    # output them
    return combdata, labl


def save(dt,lt):

    # outfile = '/media/papachristo/My Passport/finalp/MyQualysis/combdata/100Hz/'
    outfile = '/media/papachristo/My Passport/finalp/MyQualysis/combdata/ntueval/cs/'

    # dv, lv = combval()
    name = input('File name:')
    #
    # # TRAIN
    np.save(outfile + name + '.npy', np.asarray(np.float32(dt)))
    # # SAVE LABELS
    # torch.save(lt, outfile + 'train_l' + name + '.pkl')
    output = open(outfile + 'L' + name + '.pkl', 'wb')
    pickle.dump(lt, output)
    output.close()


def selntu():

    ntudir = '/media/papachristo/My Passport/finalp/NTU-RGB-D/xview/'
    ntu = np.load(ntudir + "train_data.npy")
    pkl_file = open(ntudir + 'train_label.pkl', 'rb')
    ntul = pickle.load(pkl_file)
    pkl_file.close()
    subj = [5, 6, 7, 8, 9, 10, 11]
    actions = [26, 0, 8, 9, 23, 6, 22, 2, 3, 7]
    titls = []
    lbls = []
    comblbl = []
    ntudict = dict()
    dt = []

    print(len(ntul[0]))
    for ix in range(len(ntul[0])):
        t = ntul[0][ix]  # copy title
        l = ntul[1][ix]  # copy label
        d = ntu[ix]  # copy data
        s = int(t[2:4])  # identify subject
        action = int(t[18:20]) - 1
        # if action is wanted and its less than ran
        if action in actions:
            dt.append(np.asarray(d))
            lbls.append(int(l))
            titls.append(t)
            ntudict[str(action)] = ntudict.get(str(action), 0) + 1

    # print(ntudict)
    print(sum(ntudict.values()))
    print(ntudict)
    comblbl.append(titls)
    comblbl.append(lbls)
    print(len(comblbl[0]))
    print(np.asarray(dt).shape)
    # # output them
    return dt, comblbl


# ASSIGN DIRECTORY and FILE
# comment out which function not in use, and the file name if you want

directory = '/media/papachristo/My Passport/finalp/MyQualysis/combdata/Vlf/'
file = 'csvl'
# file = input('Input file name: ')
dt, lt = selectop(file)
# dt, lt = selntu()

sve = input('Save files [y/n]? ')
if sve == 'y':
    save(dt, lt)
