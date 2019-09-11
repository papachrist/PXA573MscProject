import c3d
import numpy as np
import random

file = '/media/papachristo/My Passport/finalp/MyQualysis/output/curr/S03P1.npy'
data = np.load(file)

ds = data.shape

j = np.random.randn(30, 5)
print(j.shape)

writer = c3d.Writer()

# frame = []
for a in range(ds[0]):
    for f in range(ds[2]):
        frame = []
        for j in range(ds[3]):
            x = data[a][0][f][j][0]
            y = data[a][2][f][j][0]
            z = data[a][1][f][j][0]
            frame.append([[x, y, z, random.uniform(-1, 5), random.uniform(-1, 5)], np.asarray([])])
        # print(np.asarray(frame).shape)
        writer.add_frames(np.asarray(frame))


with open('/media/papachristo/F6C1-D2B5/Robotics/Final Project/backto/S03P1back.c3d', 'wb') as h:
    writer.write(h)