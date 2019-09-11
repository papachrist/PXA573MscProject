import c3d
import numpy as np
# with open('/media/papachristo/My Passport/finalp/CMU/armTest06_measurement.c3d', 'rb') as handle:

import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.spatial import distance

import csv
Classes = []
with open('/media/papachristo/My Passport/finalp/Classes.csv', mode='r') as infile:
    reader = csv.reader(infile)

    for rows in reader:
        k = rows
        # v = rows[1]
        Classes.append(k)



# file = '/media/papachristo/My Passport/finalp/MyQualysis/c3dTSV/grebrushteeth.c3d'
file = '/media/papachristo/My Passport/finalp/MyQualysis/c3dTSV/KotsiosSitStandup.c3d'
# file = '/media/papachristo/My Passport/finalp/MyQualysis/c3dTSV/kotwave.c3d'

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
        frame.append(avgpoints('CFWT', 'CBWT', fr, head))                               #0 - BASE OF SPINE
        frame.append(avgpoints('STRN', 'T10B', fr, head))                               #1 - MIDL OF SPINE
        frame.append(avgpoints('NECK', 'C7BB', fr, head))                               #2 - NECK
        frame.append((avgpoints('LFHD', 'RFHD', fr, head) + fr[head.index('BCHD')])/2)  #3 - HEAD
        frame.append(fr[head.index('LSHO')])                                            #4 - LEFT SHOULDER
        frame.append(fr[head.index('LELB')])                                            #5
        frame.append(avgpoints('LWRI', 'LWRO', fr, head))                               #6
        frame.append(fr[head.index('LHND')])                                            #7
        frame.append(fr[head.index('RSHO')])                                            #8
        frame.append(fr[head.index('RSHO')])                                            #9
        frame.append(fr[head.index('RELB')])                                            #10
        frame.append(avgpoints('RWRI', 'RWRO', fr, head))                               #11
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
    for fr in data:

        newfr = []
        origin = fr[1]
        for j in fr:
            newfr.append(j - origin)
        transd.append(newfr)

    return np.asarray(transd)

def rotationx(data):
    rsho = data[8]
    lsho = data[4]
    # print(rsho, lsho)
    xvec = [0, 0, 1]
    vec = rsho - lsho

    rotz = math.degrees(math.atan(vec[1] / vec[0]))
    roty = math.degrees(math.atan(vec[2] / vec[0]))

    print(rotz, roty)


data, Headers = loadata(file)




new = map(data, Headers)
print('Mapped data shape:')
print(new.shape)


timer = 100
movement = []
ta = []
cutp = []
cy = []
flag = 0
joint = 20
Labels = []

for t in range(15, 6000):
    spmov = lenbone(new[t][joint], new[t - 7][joint])
    movement.append(spmov)
    ta.append(t)
    cy.append(new[t][joint][1]/20)
    if spmov > 10 and (timer < t or len(movement) == 0) and flag == 0:
        stp = t-10
        timer = t + 100
        flag = 1
    if flag == 1 and spmov < 5:
        flag = 0
        frames = [stp, t+20]
        print("{} {}".format("Detected frames: ", frames))
        cutp.append(frames)
        test_text = input("Class number: ")
        Labels.append(int(test_text))

# print(len(movement))
# print(max(movement), min(movement), max(movement)/min(movement))
# print(cutp)

print(len(cutp))
for l in range(len(Labels)):
    print(Labels[l], Classes[Labels[l]])

plt.plot(ta, cy, 'b--', ta, movement, 'r--')
plt.show()

# with open('coors_new.csv', mode='w') as outfile:
#     writer = csv.writer(Labels)
