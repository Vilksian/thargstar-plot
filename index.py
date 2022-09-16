import numpy as np

thargstar_1 = np.array([
    [686.125, -372.875, -1832.375],     # Oochorrs UF-J c11-0
    [658.625, -384.21875, -1783.53125], # Oochorrs CS-F c13-0
    [650.46875, -382.9375, -1777.0625], # Oochorrs BS-F c13-0
    [619.25, -358.375, -1721],          # HD 38291
    [634.25, -349.9375, -1700.40625],   # Oochorrs JE-C c15-0
    [642.625, -345.5, -1676.125],       # Oochorrs SZ-N d7-5
    [620.75, -341.75, -1622.40625],     # Oochorrs PV-Y c16-0
    [599.40625, -323.25, -1610.0625],   # Oochorrs OV-Y c16-0
    [576.5625, -303.5, -1541.78125],    # Col 69 Sector JG-W c2-0
])

thargstar_2 = np.array([
    [-2016.65625, -654.6875, -2637.65625],  # Slegi GS-X b45-0
    [-2000.40625, -640.75, -2624.5625],     # Slegi JD-W b46-0
    [-1977.1875, -651.375, -2581.5625],     # Slegi SP-E d12-5
    [-1938.28125, -628.5, -2523.5],         # Slegi GI-N b51-0
    [-1893.8125, -613.1875, -2488.59375],   # Slegi KT-L b52-0
    [-1844.25, -615.59375, -2420.65625],    # Slegi DS-E b56-0
])

thargstar_3 = np.array([
    [-447.78125, -207.90625, -2110.34375],  # Oochost OI-W c4-1
    [-436.4375, -186.6875, -2038.1875],     # Oochost WU-S C6-0
    [-421.375, -183.59375, -1982.75],       # Oochost DM-P c8-0
    [-410.9375, -184.3125, -1919.03125],    # Oochost DW-T d4-7
    # [-375.28125, -177.4375, -1838.03125],   # Oochost IC-S d5-4 # Not confirmed
])


# Calculate the mean of the points, i.e. the 'center' of the cloud

thargstar_1_mean = thargstar_1.mean(axis=0)
thargstar_2_mean = thargstar_2.mean(axis=0)
thargstar_3_mean = thargstar_3.mean(axis=0)
print(thargstar_3_mean)

# Do an SVD on the mean-centered data.
t1_uu, t1_dd, t1_vv = np.linalg.svd(thargstar_1 - thargstar_1_mean)
t2_uu, t2_dd, t2_vv = np.linalg.svd(thargstar_2 - thargstar_2_mean)
t3_uu, t3_dd, t3_vv = np.linalg.svd(thargstar_3 - thargstar_3_mean)
print(t3_uu, "\n--------------\n", t3_dd, "\n-------------\n-", t3_vv, "\n--------------\n",)
# Now t1_vv[0] contains the first principal component, i.e. the direction
# vector of the 'best fit' line in the least squares sense.
# Now generate some points along this best fit line, for plotting.
# I use -7, 7 since the spread of the data is roughly 14
# and we want it to have mean 0 (like the points we did
# the svd on). Also, it's a straight line, so we only need 2 points.
thargstar_1_linepts = t1_vv[0] * np.mgrid[-100:3000:2j][:, np.newaxis]
thargstar_2_linepts = t2_vv[0] * np.mgrid[-100:3000:2j][:, np.newaxis]
thargstar_3_linepts = t3_vv[0] * np.mgrid[-100:3000:2j][:, np.newaxis]

# shift by the mean to get the line in the right place

thargstar_1_linepts += thargstar_1_mean
thargstar_2_linepts += thargstar_2_mean
thargstar_3_linepts += thargstar_3_mean

all_points = np.append(thargstar_1, [[-83.5625, -73.40625, -244.34375], [-11.5625, -15.40625, -181.09375], [76.5625, 9.34375, -183.4375], [0, 0, 0], [-41.3125, -58.96875, -354.78125]], axis=0)
all_points = np.append(all_points, thargstar_2, axis=0)
all_points = np.append(all_points, thargstar_3, axis=0)
# Verify that everything looks right.

print(all_points)

import matplotlib.pyplot as plt

import mpl_toolkits.mplot3d as m3d


ax = m3d.Axes3D(plt.figure())
ax.scatter3D(*all_points.T)

ax.plot3D(*thargstar_1_linepts.T)
ax.plot3D(*thargstar_2_linepts.T)
ax.plot3D(*thargstar_3_linepts.T)

plt.show()