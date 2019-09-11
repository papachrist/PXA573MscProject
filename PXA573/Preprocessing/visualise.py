import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.spatial import distance

data = np.load("/media/papachristo/My Passport/finalp/NTU-RGB-D/xsub/val_data.npy")


def lenbone(a, b):
    dist = distance.euclidean(a, b)
    return dist

print((data).shape)
print(data[0][0][0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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
hb = [[4,3], [3,21], [21,2], [2,1], [17,18], [18,19]]
hb = np.array(hb) - 1
# [action][x,y,z][time][specific joint][?]
action = 226
t = 0

joint = []

for s in range(25):
    x = data[action][0][t][s][0]
    y = data[action][1][t][s][0]
    z = data[action][2][t][s][0]
    joint.append([x, y, z])

joint = np.array(joint)

print('parallel:')
print(joint[8]-joint[4])
print(joint[0]-joint[20])



for j in joint:
    ax.scatter(j[0], j[1], j[2])

for b in bones:
     line, = ax.plot([joint[b[0]][0], joint[b[1]][0]], [joint[b[0]][1], joint[b[1]][1]], [joint[b[0]][2], joint[b[1]][2]])

height = 0
for hh in hb:
    l = lenbone(joint[hh[0]], joint[hh[1]])
    height = height + l

print(joint[4])
print(joint[8])


print(height)
print(lenbone(joint[8], joint[4]))
plt.show()