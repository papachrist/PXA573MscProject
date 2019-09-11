import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import matplotlib.animation as animation
import torch
from torch import IntTensor

# f1 = '/media/papachristo/My Passport/finalp/NTU-RGB-D/mydata/val_l1'
# f2 = '/media/papachristo/My Passport/finalp/NTU-RGB-D/mydata/train_l1'
# filename = "/media/papachristo/My Passport/finalp/NTU-RGB-D/xsub/val_label.pkl"


'''author: PXA573, Data processor, receives c3d motion capture
 files created on Qualysis, with markers placed in the configuration
 chosen by the author.
  The markers have to be labelled with their codes as demonstrated in
  the map function.
  It maps the file to the NTU RGB+D skeleton structure and converts the
  data format so it can be saved as output. It contains automated detection
  of events, as well as manual.
  '''

import c3d
import numpy as np
from scipy.spatial import distance
import math
import matplotlib.pyplot as plt
import pickle
import torch
from matplotlib.widgets import SpanSelector
import os

direct = '/media/papachristo/My Passport/finalp/MyQualysis/c3dTSV/'
outfile = '/media/papachristo/My Passport/finalp/MyQualysis/100Hz/'

def lenbone(a, b):
    dist = distance.euclidean(a, b)
    return dist

def loadfiles(directory):
    fl = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                fl.append(file.replace('.pkl', ''))
    return fl

def loadata(file):

    with open(file, 'rb') as handle:
        # print(c3dCONV.loadC3D(handle))

        reader = c3d.Reader(handle)
        Data = []
        Headers = []
        unwanted = []

        # for label, frameData in enumerate(points[:, 0:3]):
        for l in reader.point_labels:
            Labels = l.strip(' ') #{l.strip('acrobatwednesday2:').strip(' ')}
            Headers.append(Labels)

        for i, (frame, points, f) in enumerate(reader.read_frames()):
            frame = []
            for joint, pose in enumerate(points[:, 0:3]):
                frame.append(pose)
            Data.append(frame)
        data = np.asarray(Data)
    return data, Headers


def selectfr(st, end, data):

    cutd = []
    for i, fr in enumerate(data):
        if i >= st and i < end:
            cutd.append(fr)

    return np.asarray(cutd)


def avgpoints(a, b, data, head):
    mata = data[head.index(a)]
    matb = data[head.index(b)]
    new = (mata + matb)/2
    return new


def map(data, head):
    new = []
    for fr in data:
        frame = []
        frame.append(avgpoints('CFWT', 'CBWT', fr, head))                               #0 - BASE OF SPINE
        frame.append(avgpoints('STRN', 'T10B', fr, head))                               #1 - MIDL OF SPINE
        frame.append(0.6*(fr[head.index('NECK')])+ 0.4*(fr[head.index('BCHD')]))        #2 - NECK
        frame.append((avgpoints('LFHD', 'RFHD', fr, head) + fr[head.index('BCHD')])/2)  #3 - HEAD
        frame.append(fr[head.index('LSHO')])                                            #4 - LEFT SHOULDER
        frame.append(fr[head.index('LELB')])                                            #5
        frame.append(avgpoints('LWRI', 'LWRO', fr, head))                               #6
        frame.append(fr[head.index('LHND')])                                            #7
        frame.append(fr[head.index('RSHO')])                                            #8
        frame.append(fr[head.index('RELB')])                                            #9
        frame.append(avgpoints('RWRI', 'RWRO', fr, head))                               #10
        frame.append(fr[head.index('RHND')])                                            #11
        frame.append(fr[head.index('LFWT')])                                            #12
        frame.append(fr[head.index('LKNE')])                                            #13
        frame.append(fr[head.index('LANK')])                                            #14
        frame.append(avgpoints('LMT5', 'LTOE', fr, head))                               #15
        frame.append(fr[head.index('RFWT')])                                            #16
        frame.append(fr[head.index('RKNE')])                                            #17
        frame.append(fr[head.index('RANK')])                                            #18
        frame.append(avgpoints('RMT5', 'RTOE', fr, head))                               #19
        frame.append(avgpoints('CLAV', 'C7BB', fr, head))                               #20
        frame.append(fr[head.index('LTFI')])                                            #21
        frame.append(fr[head.index('LTHM')])                                            #22
        frame.append(fr[head.index('RTFI')])                                            #23
        frame.append(fr[head.index('RTHM')])                                            #24


        new.append(frame)

    return np.asarray(new)


def transtorigin(data):
    transd = []
    origin = data[0][1] #- [0, 0, 2.5]
    for fr in data:

        newfr = []

        for j in fr:
            s = j - origin
            s[1] = s[1] + 3000
            newfr.append(s)
        transd.append(newfr)

    return np.asarray(transd)


def convertoNTU(data, frate):
    counter = 0
    fx = []
    fy = []
    fz = []
    ntu = []
    # print(reader.point_labels, Headers)

    for i in range(900):

        counter +=1
        jx = []
        jy = []
        jz = []
        if counter == frate:
            counter = 0
            if i < data.shape[0]:
                fr = data[i]
                for frDt in fr:
                    jx.append([frDt[0]/1000, 0])
                    jy.append([frDt[2]/1000, 0])
                    jz.append([frDt[1]/1000, 0])
                    # print(Headers[joint],frDt)
                    # print(frame)
                fx.append(list(jx))
                fy.append(list(jy))
                fz.append(list(jz))
            else:
                for s in range(25):
                    jx.append([0, 0])
                    jy.append([0, 0])
                    jz.append([0, 0])
                fx.append(list(jx))
                fy.append(list(jy))
                fz.append(list(jz))
    # print(len(fx))
    ntu.append(fx)
    ntu.append(fy)
    ntu.append(fz)
    ntu = np.asarray(ntu)

    # print(ntu.shape)
    #     # map cmu to st-gcn base
    #     # first identify index

    return ntu


def rotationx(data):
    rsho = data[8]
    lsho = data[4]
    # print(rsho, lsho)
    xvec = [0, 0, 1]
    vec = rsho - lsho

    rotz = math.degrees(math.atan(vec[1] / vec[0]))
    roty = math.degrees(math.atan(vec[2] / vec[0]))

    print(rotz, roty)


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


def onselect(xmin, xmax):
    # function for plot select

    print('{} {}'.format('Selected: ', [xmin, xmax]))

    cl = input("Class number (if wrong, type w): ")
    if cl != 'w':
        x.append([xmin, xmax])
        l.append(int(cl))


def runevents():
    classes = readClasses()

    sn = input("Subject number: ")
    subject = 'S' + str(sn).zfill(2)


    stp = 0
    timer = 100
    movement = []
    ta = []
    cutp = []
    cy = []
    flag = 0
    Labels = []
    titls = []
    clabel = []

    for i in range(len(cut)):
        cd = cut[i]
        cl = lbl[i]
        cutp.append(cd)
        title = subject + 'F' + str(cd[0]).zfill(4) + str(cd[1]).zfill(4) + 'A' + str(cl)
        titls.append(title)
        clabel.append(int(cl))



    Labels.append(titls)
    Labels.append(clabel)

    print("{} {}".format("Number of events: ", len(cutp)))

    return cutp, Labels


def runtraj(file, frate):
    fdata = []
    data, Headers = loadata(file)

    # print(Headers)
    print('Input data shape:')
    print(data.shape)

    new = map(data, Headers)
    print('Skeleton mapping complete')
    # print(new.shape)

    cutp, labels = runevents()

    for ind in range(len(cut)):
        fr = cut[ind]
        selfr = selectfr(fr[0], fr[1], new)
        # print('Selected data shape:')
        # print(selfr.shape)
        trd = transtorigin(selfr)
        # print('Translation to Origin complete')
        # print(trd.shape)
        ntu = convertoNTU(trd, frate)
        # print('NTU structure complete')
        fdata.append(ntu)

    print('Data preprocessing complete')
    return np.asarray(fdata), labels

# main program:
def main(file):
    # frate = int(input('Select Frame rate: '))
    frate = 1
    data, labels = runtraj(file, frate)

    print(data.shape)
    sve = input('save? [y/n]')
    if sve == 'y':
        name = input('File name:')
        np.save(outfile + name, data)

        #SAVE LABELS
        output = open(outfile + 'L' + name + '.pkl', 'wb')
        pickle.dump(labels, output)
        output.close()


directory = '/media/papachristo/My Passport/finalp/MyQualysis/output/'


fls = loadfiles(directory)
for lfile in fls:
    fl = directory + lfile + '.pkl'
    file = open(fl, 'rb')
    labels = pickle.load(file)
    c = 0
    cut = []
    lbl = []
    cntr = 0
    unwanted = []
    for l in labels[0]:
        print(l)
    for ind in range(len(labels[0])):
            # k = d.int()
            # print(type(k))
        d = labels[0][ind]
        lb = labels[1][ind]
        if lb > 9:
            if len(d) == 15:
                cut.append([int(d[4:8]), int(d[8:12])])
                # print(lb)
                lbl.append(lb)
        else:
            if len(d) == 14:
                cut.append([int(d[4:8]), int(d[8:12])])
                # print(lb)
                lbl.append(lb)
            # print(int(d[4:8]), int(d[8:12]))
        cntr += 1
    x = []
    l = []
    print(lfile)
    doit = input('Alter? [y/n]')
    if doit == 'y':
        dfile = direct + input('Enter Data file: ') + '.c3d'
        main(dfile)

# CREATE automatic selection
# PAIR SELECTION PREPROCESSING