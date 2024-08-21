import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 14})
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
# CVD accessible colors
# black - dark red - indigo (blueish) - yellow - teal (greenish) - light gray
colors = ['#000000','#c1272d', '#0000a7', '#eecc16', '#008176', '#b3b3b3']
# black - dark grey - grey - light grey - very light grey
greyscale = ['#000000', '#333333', '#666666', '#999999', '#cccccc']

################################################################################
# READ HPX FILES
# get header and loop number for averaging
hpx_header = np.genfromtxt(os.path.abspath('./data_hpx/cores_hpx_right_200.txt'), dtype='unicode', delimiter=';' , max_rows=1)
n_loop_hpx = int(hpx_header[-1])

# read tiles file for 128 cores
hpx_tiles_right_128_matrix = np.genfromtxt(os.path.abspath('./data_hpx/tiles_hpx_right_128.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_hpx_tiles = int(hpx_tiles_right_128_matrix.shape[0]/n_loop_hpx)
hpx_tiles_right_128_averaged = np.zeros((n_entries_hpx_tiles, hpx_tiles_right_128_matrix.shape[1] - 2))
for i in range (n_entries_hpx_tiles):
    hpx_tiles_right_128_averaged[i,:] = np.mean(hpx_tiles_right_128_matrix[i*n_loop_hpx:(i+1)*n_loop_hpx,2:],axis=0)
hpx_tiles_right_128_averaged[:,4] = hpx_tiles_right_128_averaged[:,4] / 1000000.0
# careful only total total time ist converted to seconds

# read tiles file for 16 cores
hpx_tiles_right_16_matrix = np.genfromtxt(os.path.abspath('./data_hpx/tiles_hpx_right_16.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_hpx_tiles = int(hpx_tiles_right_16_matrix.shape[0]/n_loop_hpx)
hpx_tiles_right_16_averaged = np.zeros((n_entries_hpx_tiles, hpx_tiles_right_16_matrix.shape[1] - 2))
for i in range (n_entries_hpx_tiles):
    hpx_tiles_right_16_averaged[i,:] = np.mean(hpx_tiles_right_16_matrix[i*n_loop_hpx:(i+1)*n_loop_hpx,2:],axis=0)
hpx_tiles_right_16_averaged[:,4] = hpx_tiles_right_16_averaged[:,4] / 1000000.0
# careful only total total time ist converted to seconds

# read tiles file for 18 cores CPU only
hpx_tiles_right_18_cpu_matrix = np.genfromtxt(os.path.abspath('./data_hpx/tiles_hpx_right_cpu_18.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_hpx_tiles = int(hpx_tiles_right_18_cpu_matrix.shape[0]/n_loop_hpx)
hpx_tiles_right_18_cpu_averaged = np.zeros((n_entries_hpx_tiles, hpx_tiles_right_18_cpu_matrix.shape[1] - 2))
for i in range (n_entries_hpx_tiles):
    hpx_tiles_right_18_cpu_averaged[i,:] = np.mean(hpx_tiles_right_18_cpu_matrix[i*n_loop_hpx:(i+1)*n_loop_hpx,2:],axis=0)
hpx_tiles_right_18_cpu_averaged[:,4] = hpx_tiles_right_18_cpu_averaged[:,4] / 1000000.0
# careful only total total time ist converted to seconds

# read tiles file for 18 cores CPU plus GPU
hpx_tiles_right_18_gpu_matrix = np.genfromtxt(os.path.abspath('./data_hpx/tiles_hpx_right_gpu_18.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_hpx_tiles = int(hpx_tiles_right_18_gpu_matrix.shape[0]/n_loop_hpx)
hpx_tiles_right_18_gpu_averaged = np.zeros((n_entries_hpx_tiles, hpx_tiles_right_18_gpu_matrix.shape[1] - 2))
for i in range (n_entries_hpx_tiles):
    hpx_tiles_right_18_gpu_averaged[i,:] = np.mean(hpx_tiles_right_18_gpu_matrix[i*n_loop_hpx:(i+1)*n_loop_hpx,2:],axis=0)
hpx_tiles_right_18_gpu_averaged[:,4] = hpx_tiles_right_18_gpu_averaged[:,4] / 1000000.0
# careful only total total time ist converted to seconds


# read tiles file for 18 cores CPU plus GPU wirh recycled memory
hpx_tiles_gpu_recycle_averaged = np.genfromtxt(os.path.abspath('./data_hpx/tiles_gpu_18_recycle.txt'), dtype='float', delimiter=';' , skip_header=1)
hpx_tiles_gpu_recycle_averaged[:,6] = hpx_tiles_gpu_recycle_averaged[:,6] / 1000000.0


# # read tiles file for 128 cores
# hpx_tiles_cpu_128_mkl = np.genfromtxt(os.path.abspath('./data_hpx/tiles_cpu_128_mkl.txt'), dtype='float', delimiter=';' , skip_header=1)
# hpx_tiles_cpu_128_mkl[:,6] = hpx_tiles_cpu_128_mkl[:,6] / 1000000.0

# # read tiles file for 128 cores
# hpx_tiles_cpu_16_mkl = np.genfromtxt(os.path.abspath('./data_hpx/tiles_cpu_16_mkl.txt'), dtype='float', delimiter=';' , skip_header=1)
# hpx_tiles_cpu_16_mkl[:,6] = hpx_tiles_cpu_16_mkl[:,6] / 1000000.0


# # read tiles file for 18 cores intel
# hpx_tiles_cpu_18_mkl = np.genfromtxt(os.path.abspath('./data_hpx/tiles_cpu_18_mkl.txt'), dtype='float', delimiter=';' , skip_header=1)
# hpx_tiles_cpu_18_mkl[:,6] = hpx_tiles_cpu_18_mkl[:,6] / 1000000.0


# # HPX TILE SCALING
# # plot HPX Runtime distribution tiles_right
# points = hpx_tiles_right_128_averaged[:,0]
# plt.figure(figsize=(10,5))
# plt.plot(points, hpx_tiles_right_16_averaged[:,4], 'o-', c=colors[1], linewidth=1, label='System 1, 16 Cores')
# plt.plot(points, hpx_tiles_right_128_averaged[:,4], 'o-', c=colors[0], linewidth=1, label='System 1, 128 Cores')
# plt.plot(points, hpx_tiles_cpu_16_mkl[:,6], 'o--', c=colors[1], linewidth=1, label='System 1, 16 Cores MKL')
# plt.plot(points[:-1], hpx_tiles_cpu_128_mkl[:,6], 'o--', c=colors[0], linewidth=1, label='System 1, 128 Cores MKL')
# plt.plot(points, hpx_tiles_right_18_cpu_averaged[:,4], 's--', c=colors[2], linewidth=1, label='System 2, 18 Cores')
# plt.plot(points, hpx_tiles_cpu_18_mkl[:,6], 's-', c=colors[2], linewidth=1, label='System 2, 18 Cores MKL')
# plt.plot(points, hpx_tiles_right_18_gpu_averaged[:,4], 's--', c=colors[4], linewidth=1, label='System 2, 18 Cores + GPU')
# plt.plot(points[2:], hpx_tiles_gpu_recycle_averaged[:,6], 's--', c=colors[3], linewidth=1, label='System 2, 18 Cores + GPU recycle')
# #plt.title('Tile scaling HPX implementation for different tile sizes')
# plt.legend(loc='upper right')
# plt.xlabel('Tile size and tiles per dimension ')
# plt.xscale("log")
# labels_x = (20000 / points).astype(int).astype(str)
# for i in range(0,labels_x.size):
#     labels_x[i] = labels_x[i] + "\n T = " + points[i].astype(int).astype(str)
# plt.xticks(ticks=points, labels=labels_x)
# plt.yscale("log")
# plt.ylabel('Runtime in s')
# plt.savefig('figures/tiles_right_scaling_mkl.pdf', bbox_inches='tight')


plt.figure(figsize=(7,5))
plt.grid()
points = hpx_tiles_right_128_averaged[1:,0]
# error bars
plt.plot(points, hpx_tiles_right_128_averaged[1:,4], 'o-', c=colors[1], linewidth=2, label='System 1, 128 Cores')
plt.plot(points, hpx_tiles_right_18_cpu_averaged[1:,4], 's-', c=colors[2], linewidth=2, label='System 2, 18 Cores')
plt.plot(points, hpx_tiles_right_18_gpu_averaged[1:,4], 's-', c=colors[4], linewidth=2, label='System 2, 18 Cores + GPU')
#plt.plot(points[2:], hpx_tiles_gpu_recycle_averaged[:,6], 's--', c=colors[3], linewidth=1, label='System 2, 18 Cores + GPU recycle')
# plot parameters
plt.legend(bbox_to_anchor=(.8, 1), loc='upper right')
plt.xlabel('Tiles per dimension ')
plt.xscale("log")
plt.xticks(ticks=points, labels=points.astype(int).astype(str))
plt.yscale("log")
plt.ylabel('Runtime in s')
plt.savefig('figures/tiles_right_scaling_reframe.pdf', bbox_inches='tight')