
import numpy as np
import pickle
import random
import math
import os

def loadfiles(directory):
    fl = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                fl.append(file.replace('.npy', ''))
    return fl


def comb(fl):
    combdata = []
    comblabl = []
    combtitl = []
    labl = []

    for f in fl:
        data = directory + f + '.npy'
        Labels = directory + 'L' + f + '.pkl'

        # loader = self.data_loader['test']
        pkl_file = open(Labels, 'rb')
        labels = pickle.load(pkl_file)
        pkl_file.close()

        valdata = np.load(data)
        # print(len(valdata))
        for ind in range(len(valdata)):
            combdata.append(valdata[ind])
            comblabl.append(int(labels[1][ind]))
            combtitl.append(labels[0][ind])

    labl.append(combtitl)
    labl.append(comblabl)

    # output them
    return combdata, labl


def combntu(actions, ntudict, dt, lt, ran, tr):

    ntudir = '/media/papachristo/My Passport/finalp/NTU-RGB-D/xview/'

    if tr:
        ntu = np.load(ntudir + "train_data.npy")
        pkl_file = open(ntudir + 'train_label.pkl', 'rb')
        ntul = pickle.load(pkl_file)
        pkl_file.close()
        subj = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31]
    else:
        ntu = np.load(ntudir + "val_data.npy")
        pkl_file = open(ntudir + 'val_label.pkl', 'rb')
        ntul = pickle.load(pkl_file)
        pkl_file.close()
        subj = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 34]
    titls = lt[0]
    lbls = lt[1]
    comblbl = []
    for ix in range(len(ntul[0])):
        t = ntul[0][ix] # copy title
        l = ntul[1][ix] # copy label
        d = ntu[ix]     # copy data
        s = int(t[10:12]) # identify subject
        # if s not in subj:
        action = int(t[18:20]) - 1
        # if action is wanted and its less than ran
        if action in actions and ntudict.get(str(action), 0) < ran:
            dt.append(np.asarray(d))
            lbls.append(int(l))
            t1 = t.replace('.skeleton', '')
            rand = math.ceil(random.uniform(0,5000))
            tf = t1[0:2] + str(s) + 'F' + str(rand).zfill(4) + str(rand + 219).zfill(4) + 'A' + str(action)
            titls.append(tf)
            ntudict[str(action)] = ntudict.get(str(action), 0) + 1

    # print(ntudict)
    print(sum(ntudict.values()))

    comblbl.append(titls)
    comblbl.append(lbls)
    #
    # # output them
    return dt, comblbl


def save(dt, lt, name, Tr):
    if Tr == 0:
        outfile = '/media/papachristo/My Passport/finalp/MyQualysis/combdata/Trf/'
    else:
        outfile = '/media/papachristo/My Passport/finalp/MyQualysis/combdata/Vlf/'
    # dv, lv = combval()

    #
    # # TRAIN
    np.save(outfile + name + '.npy', np.asarray(np.float32(dt)))
    # # SAVE LABELS
    # torch.save(lt, outfile + 'train_l' + name + '.pkl')
    output = open(outfile + 'L' + name + '.pkl', 'wb')
    pickle.dump(lt, output)
    output.close()


def identactions(lt):
    acdict = dict()
    actions = []
    for l in lt[1]:
        if l not in actions:
            actions.append(l)
            acdict[str(l)] = 1
        elif len(actions) != 0:
            acdict[str(l)] = acdict.get(str(l), 0) + 1

    return actions, acdict


def splitvt(d, l, actions):
    sdt, stt, slt = [], [], []
    sdv, stv, slv = [], [], []
    combt, combv = [], []

    # split to actions
    for ac in actions:
        acd, acl, act = [], [], []
        for ix in range(len(d)):
            # identify specific action
            if l[1][ix] == ac:
                acd.append(d[ix])
                act.append(l[0][ix])
                acl.append(l[1][ix])

        # shuffle data and labels together using their index
        arind = list(range(0,len(acd)))
        random.shuffle(arind)
        thresh = int(round(0.7 * len(arind)))
        trix = arind[:thresh]
        vix = arind[thresh:]
        # print(len(arind), len(trix), len(vix))

        # separate data&labels into train and validation
        for ind in range(len(arind)):
            if ind in trix:
                sdt.append(acd[ind])
                stt.append(act[ind])
                slt.append(acl[ind])
            elif ind in vix:
                sdv.append(acd[ind])
                stv.append(act[ind])
                slv.append(acl[ind])
        # print(len(sdt), len(sdv))
    # title with label
    combt.append(stt)
    combt.append(slt)
    combv.append(stv)
    combv.append(slv)

    sa = input('Save? [y/n] ')

    if sa == 'y':
        name = input('File name:')
        print('Tr')
        save(sdt, combt, name, 0)
        print('Vl')
        save(sdv, combv, name, 1)


# main code
directory = '/media/papachristo/My Passport/finalp/MyQualysis/combdata/newtr/'
fl = loadfiles(directory)
# 1. COMBINE DATA

# dt, lt = comb(['VL2'])


for f in fl:
    print(f)
    data = directory + f + '.npy'
    Labels = directory + 'L' + f + '.pkl'
    dt = np.load(data)
    # loader = self.data_loader['test']
    pkl_file = open(Labels, 'rb')
    lt = pickle.load(pkl_file)
    pkl_file.close()

    actions, acdict = identactions(lt)
    print(acdict)
    print(sum(acdict.values()))


    splitvt(dt, lt, actions)
