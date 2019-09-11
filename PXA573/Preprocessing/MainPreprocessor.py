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

#Directories
direct = '/media/papachristo/My Passport/finalp/MyQualysis/c3dTSV/'
outfile = '/media/papachristo/My Passport/finalp/MyQualysis/output/'
outdir = '/media/papachristo/My Passport/finalp/MyQualysis/output/'

def lenbone(a, b):
    dist = distance.euclidean(a, b)
    return dist


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
        frame.append(avgpoints('CFWT', 'CBWT', fr, head))                               # 0  - BASE OF SPINE
        frame.append(avgpoints('STRN', 'T10B', fr, head))                               # 1  - MIDL OF SPINE
        frame.append(0.6*(fr[head.index('NECK')])+ 0.4*(fr[head.index('BCHD')]))        # 2  - NECK
        frame.append((avgpoints('LFHD', 'RFHD', fr, head) + fr[head.index('BCHD')])/2)  # 3  - HEAD
        frame.append(fr[head.index('LSHO')])                                            # 4  - LEFT SHOULDER
        frame.append(fr[head.index('LELB')])                                            # 5  - LEFT ELBOW
        frame.append(avgpoints('LWRI', 'LWRO', fr, head))                               # 6  - LEFT WRIST
        frame.append(fr[head.index('LHND')])                                            # 7  - LEFT HAND
        frame.append(fr[head.index('RSHO')])                                            # 8  - RIGHT SHOULDER
        frame.append(fr[head.index('RELB')])                                            # 9  - RIGHT ELBOW
        frame.append(avgpoints('RWRI', 'RWRO', fr, head))                               # 10 - RIGHT WRIST
        frame.append(fr[head.index('RHND')])                                            # 11 - RIGHT HAND
        frame.append(fr[head.index('LFWT')])                                            # 12 - LEFT WAIST
        frame.append(fr[head.index('LKNE')])                                            # 13 - LEFT KNEE
        frame.append(fr[head.index('LANK')])                                            # 14 - LEFT ANKLE
        frame.append(avgpoints('LMT5', 'LTOE', fr, head))                               # 15 - LEFT FOOT
        frame.append(fr[head.index('RFWT')])                                            # 16 - RIGHT WAIST
        frame.append(fr[head.index('RKNE')])                                            # 17 - RIGHT KNEE
        frame.append(fr[head.index('RANK')])                                            # 18 - RIGHT ANKLE
        frame.append(avgpoints('RMT5', 'RTOE', fr, head))                               # 19 - RIGHT FOOT
        frame.append(avgpoints('CLAV', 'C7BB', fr, head))                               # 20 - CLAVICLE
        frame.append(fr[head.index('LTFI')])                                            # 21 - LEFT TOP FINGERS
        frame.append(fr[head.index('LTHM')])                                            # 22 - LEFT THUMB
        frame.append(fr[head.index('RTFI')])                                            # 23 - RIGHT TOP FINGERS
        frame.append(fr[head.index('RTHM')])                                            # 24 - RIGHT THUMB

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


def convertoNTU(data):
    counter = 0
    fx = []
    fy = []
    fz = []
    ntu = []
    # print(reader.point_labels, Headers)
    frate = 3
    for i in range(300*frate):

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


def runevents(new, auto = True):
    classes = readClasses()

    sn = input("Subject number: ")
    subject = 'S' + str(sn).zfill(2)


    stp = 0
    timer = 100
    movement = []
    ta = []
    cutp = []
    jmv = []
    jmv2 = []
    flag = 0
    Labels = []
    titls = []
    clabel = []

    plot = input('Use plot selection [y/n]? ')
    if plot != 'y':
        if auto:
            joint = int(input('Which Joint to use: '))
            cont = input('Is it continuous [y/n]? ')
            if cont != 'y':
                for t in range(15, new.shape[0]):
                    spmov = lenbone(new[t][joint], new[t - 7][joint])
                    movement.append(spmov)
                    ta.append(t)
                    jmv.append(new[t][joint][1] / 20)
                    if spmov > 15 and (timer < t or len(movement) == 0) and flag == 0:
                        stp = t - 15
                        timer = t + 100
                        flag = 1
                    if flag == 1 and spmov < 5:
                        flag = 0
                        end = t + 10
                        frames = [stp, end]
                        print("{} {}".format("Detected frames: ", frames))
                        cl = input("Class number(If wrong, type w!): ")
                        if cl != 'w':
                            cutp.append(frames)
                            title = subject + 'F' + str(stp).zfill(4) + str(end).zfill(4) + 'A' + cl
                            titls.append(title)
                            clabel.append(int(cl))
            else:

                for t in range(15, new.shape[0], 90):
                    spmov = lenbone(new[t][joint], new[t - 7][joint])
                    movement.append(spmov)
                    ta.append(t)
                    jmv.append(new[t][joint][1] / 20)
                    if spmov > 20 and (t+250) < new.shape[0]:
                        if lenbone(new[t+250][joint], new[t + 243][joint]) > 20:
                            stp = t - 7
                            end = stp + 250
                            frames = [stp, end]
                            print("{} {}".format("Detected frames: ", frames))
                            cl = input("Class number(If wrong, type w!): ")
                            if cl != 'w':
                                cutp.append(frames)
                                title = subject + 'F' + str(stp).zfill(4) + str(end).zfill(4) + 'A' + cl
                                titls.append(title)
                                clabel.append(int(cl))
                        elif lenbone(new[t+220][joint], new[t + 213][joint]) > 20:
                            stp = t - 722

                            end = stp + 220
                            frames = [stp, end]
                            print("{} {}".format("Detected frames: ", frames))
                            cl = input("Class number(If wrong, type w!): ")
                            if cl != 'w':
                                cutp.append(frames)
                                title = subject + 'F' + str(stp).zfill(4) + str(end).zfill(4) + 'A' + cl
                                titls.append(title)
                                clabel.append(int(cl))

        else:

            endflag = True
            print('Type f, when finished!')
            while endflag:

                    stp = input("Start Frame: ")
                    if stp != 'f':
                        end = input("End Frame: ")
                        frames = [int(stp), int(end)]
                        if end != 'f':
                            cutp.append(frames)
                            cl = input("Class number: ")
                            title = subject + 'F' + str(stp).zfill(4) + str(end).zfill(4) + 'A' + cl
                            titls.append(title)
                            clabel.append(int(cl))
                        else:
                            endflag = False
                    else:
                        endflag = False

    else:
        joints = []
        select = 'm'
        while select != 'f':
            if len(joints) == 0:
                # print('Average Acceleration and first joint trajectory appear. Type f when done!')
                print('Which trajectories to show, type f when done')
            select = input('Enter joints: ')
            if select != 'f':
                joints.append(int(select))

        arr = []
        for k in range(new.shape[0]):
            arr.append(new[k][joints[0]][2])
        minarr = min(arr)
        axis = int(input('Which axis to display [x:0, y:2, z:1]? '))
        for t in range(0, new.shape[0]):
            spmov = 0
            for joint in joints:
                spmov = spmov + lenbone(new[t][joint], new[t - 7][joint])

            movement.append((spmov/len(joints))+minarr)
            ta.append(t)
            jmv.append(new[t][joints[0]][axis])
            if len(joints) > 1:
                jmv2.append(new[t][joints[1]][axis])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        if len(joints) == 1:
            ax.plot(ta, jmv, 'b--', ta, movement, 'r--')
        if len(joints) == 4:
            ax.plot(ta, jmv, 'b--', ta, movement, 'r--')
        else:
            ax.plot(ta, jmv, 'b--', ta, jmv2, 'g--', ta, movement, 'r--')
        # ax.plot(ta, cy, 'b--')
        ax.set_title('Press left mouse button and drag to select, close when finished')

        # set useblit True on gtkagg for enhanced performance
        span = SpanSelector(ax, onselect, 'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))
        plt.show()
        print(x, l)
        cutp = x

        for i in range(len(cutp)):
            title = subject + 'F' + str(math.ceil(cutp[i][0])).zfill(4) + str(math.ceil(cutp[i][1])).zfill(4) + 'A' + str(l[i])
            titls.append(title)

        for la in l:
            clabel.append(la)

    Labels.append(titls)
    Labels.append(clabel)

    print("{} {}".format("Number of events: ", len(cutp)))

    return cutp, Labels


def runtraj(file):
    fdata = []
    data, Headers = loadata(file)

    # print(Headers)
    print('Input data shape:')
    print(data.shape)

    new = map(data, Headers)
    print('Skeleton mapping complete')
    # print(new.shape)


    # ASK IF AUTO
    # DETECT EVENTS:
    mora = input("Select events Manually or Automatically? [m/a]: ")

    if mora == 'm':
        cutp, labels = runevents(new, auto = False)
    else:
        cutp, labels = runevents(new)

    for ind in range(len(cutp)):
        fr = cutp[ind]
        selfr = selectfr(fr[0], fr[1], new)
        # print('Selected data shape:')
        # print(selfr.shape)
        trd = transtorigin(selfr)
        # print('Translation to Origin complete')
        # print(trd.shape)
        ntu = convertoNTU(trd)
        # print('NTU structure complete')
        fdata.append(ntu)

    print('Data preprocessing complete')
    return np.asarray(fdata), labels

# main program:
def main(file):
    data, labels = runtraj(file)

    print(data.shape)
    name = input('File name:')
    np.save(outfile + name, data)

    #SAVE LABELS
    output = open(outdir +'L' + name + '.pkl', 'wb')
    pickle.dump(labels, output)
    output.close()


x = []
l = []
file = direct + input('Enter file: ') +'.c3d'
main(file)

# PAIR SELECTION PREPROCESSING



