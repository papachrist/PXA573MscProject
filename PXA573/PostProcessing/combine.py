import argparse
import yaml
import numpy as np
# from processor.Qualysis import readClasses
# torch
import torch
import pickle

def convert(t, mix = False):

    direct = '/media/papachristo/My Passport/finalp/MyQualysis/Out/'

    combdata = []
    comblabl = []
    combtitl = []
    loadata = []
    labl = []

    # for t in trtit:
    data = direct + t + '.npy'
    Labels = direct + 'L' + t

    # loader = self.data_loader['test']
    pkl_file = open(Labels, 'rb')
    labels = pickle.load(pkl_file)
    pkl_file.close()

    valdata = np.load(data)
    print(len(valdata))
    for ind in range(len(valdata)):
        combdata.append(valdata[ind])
        comblabl.append(int(labels[1][ind].data.tolist()[0]))
        combtitl.append(labels[0][ind])

    # NTU mix
    if mix:
        ntudir = '/media/papachristo/My Passport/finalp/NTU-RGB-D/xsub/'
        ntu = np.load(ntudir + "train_data.npy")
        pkl_file = open(ntudir + 'train_label.pkl', 'rb')
        ntul = pickle.load(pkl_file)
        pkl_file.close()
        cntr = 0
        for d in ntu:
            if cntr < 100:
                combdata.append(np.asarray(d))
                comblabl.append(ntul[1][cntr])
                combtitl.append(ntul[0][cntr])
            cntr += 1

    labl.append(combtitl)
    labl.append(comblabl)

    # output them
    return combdata, labl

def combtr(trtit, mix = False):

    direct = '/media/papachristo/My Passport/finalp/MyQualysis/Out/'

    trtit = ['S01P2', 'S01P3', 'S02P1', 'S02P3', 'S02P5', 'S01P1', 'S01P4', 'S02P2', 'S02P4']

    combdata = []
    comblabl = []
    combtitl = []
    loadata = []
    labl = []

    for t in trtit:
        data = direct + t + '.npy'
        Labels = direct + 'L' + t

        # loader = self.data_loader['test']
        pkl_file = open(Labels, 'rb')
        labels = pickle.load(pkl_file)
        pkl_file.close()

        valdata = np.load(data)
        print(len(valdata))
        for ind in range(len(valdata)):
            combdata.append(valdata[ind])
            comblabl.append(int(labels[1][ind].data.tolist()[0]))
            combtitl.append(labels[0][ind])
            # include batch

    # NTU mix
    if mix:
        ntudir = '/media/papachristo/My Passport/finalp/NTU-RGB-D/xsub/'
        ntu = np.load(ntudir + "train_data.npy")
        pkl_file = open(ntudir + 'train_label.pkl', 'rb')
        ntul = pickle.load(pkl_file)
        pkl_file.close()
        cntr = 0
        for d in ntu:
            if cntr < 100:
                combdata.append(np.asarray(d))
                comblabl.append(ntul[1][cntr])
                combtitl.append(ntul[0][cntr])
            cntr += 1

    labl.append(combtitl)
    labl.append(comblabl)

    # output them
    return combdata, labl

def batchloader(batch, data, labl):
    loadata = []
    for i in range(data.shape[0]-(batch-1)):
        if i % batch == 0:
            print(torch.from_numpy(data[i:i + 8]).shape)
            loadata.append([torch.from_numpy(data[i:i+batch]), labl[1][i:batch]])


    remain = 206 % batch
    g = batch - remain
    s = 194 - g
    print(s, s+8)

    return loadata

def combval(mix = False):

    direct = '/media/papachristo/My Passport/finalp/MyQualysis/Out/'

    valtt = ['S01P1', 'S01P4', 'S02P2', 'S02P4']

    combdata = []
    comblabl = []
    combtitl = []
    loadata = []
    labl = []

    for t in valtt:
        data = direct + t + '.npy'
        Labels = direct + 'L' + t

        # loader = self.data_loader['test']
        pkl_file = open(Labels, 'rb')
        labels = pickle.load(pkl_file)
        pkl_file.close()

        valdata = np.load(data)
        print(len(valdata))
        for ind in range(len(valdata)):
            combdata.append(valdata[ind])
            comblabl.append(int(labels[1][ind].data.tolist()[0]))
            combtitl.append(labels[0][ind])
            # include batch

    # NTU mix
    # for j in combdata:
    #     print(j)
    if mix:
        ntudir = '/media/papachristo/My Passport/finalp/NTU-RGB-D/xsub/'
        ntu = np.load(ntudir + "val_data.npy")
        pkl_file = open(ntudir + 'val_label.pkl', 'rb')
        ntul = pickle.load(pkl_file)
        pkl_file.close()
        cntr = 0
        for d in ntu:
            if cntr < 50:
                combdata.append(np.asarray(d))
                comblabl.append(ntul[1][cntr])
                combtitl.append(ntul[0][cntr])
            cntr += 1


    labl.append(combtitl)
    labl.append(comblabl)

    # output them
    return combdata, labl


titloi = ['S01P2', 'S01P3', 'S02P1', 'S02P3', 'S02P5', 'S01P1', 'S01P4', 'S02P2', 'S02P4']

outfile = '/media/papachristo/My Passport/finalp/MyQualysis/output/old/'

for t in titloi:
    dt, lt = convert(t)
    # dv, lv = combval()
    name = t
    #
    # # TRAIN
    np.save(outfile + 'old' + name + '.npy', np.asarray(np.float32(dt)))
    # # SAVE LABELS
    # torch.save(lt, outfile + 'train_l' + name + '.pkl')
    output = open(outfile + 'Lold' + name, 'wb')
    pickle.dump(lt, output)
    output.close()

# # torch.from_numpy(
# # VALIDATION
# np.save(outfile + 'val_d' + name + '.npy', np.asarray(np.float32(dv)))
# # torch.save(lv, outfile + 'val_l' + name + '.pkl')
# output = open(outfile + 'val_l' + name, 'wb')
# pickle.dump(lv, output)
# output.close()

# loader = batchloader(8, d, l)
# print(len(loader))
# for data, labels in loader:
# print(data.shape, labels.shape)