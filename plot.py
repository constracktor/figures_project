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
# READ PETSC FILES
# get header and loop number for averaging
petsc_header = np.genfromtxt(os.path.abspath('./data_petsc/cores_petsc.txt'), dtype='unicode', delimiter=';' , max_rows=1)
n_loop_petsc = int(petsc_header[-1])

# read cores file
petsc_cores_matrix = np.genfromtxt(os.path.abspath('./data_petsc/cores_petsc.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_petsc_cores = int(petsc_cores_matrix.shape[0]/n_loop_petsc)
petsc_cores_averaged = np.zeros((n_entries_petsc_cores, petsc_cores_matrix.shape[1] - 1))
for i in range (n_entries_petsc_cores):
    petsc_cores_averaged[i,:] = np.mean(petsc_cores_matrix[i*n_loop_petsc:(i+1)*n_loop_petsc,:-1],axis=0)

# read blas file
petsc_blas_matrix = np.genfromtxt(os.path.abspath('./data_petsc/blas_petsc.txt'), dtype='float', delimiter=';' , skip_header=1)

################################################################################
# READ HPX FILES
# get header and loop number for averaging
hpx_header = np.genfromtxt(os.path.abspath('./data_hpx/cores_hpx_right_200.txt'), dtype='unicode', delimiter=';' , max_rows=1)
n_loop_hpx = int(hpx_header[-1])

# read cores file double precision
hpx_cores_matrix = np.genfromtxt(os.path.abspath('./data_hpx/cores_hpx_right_200.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_hpx_data = int(hpx_cores_matrix.shape[0]/n_loop_hpx)
hpx_cores_averaged = np.zeros((n_entries_hpx_data, hpx_cores_matrix.shape[1] - 1))
for i in range (n_entries_hpx_data):
    hpx_cores_averaged[i,:] = np.mean(hpx_cores_matrix[i*n_loop_hpx:(i+1)*n_loop_hpx,1:],axis=0)
hpx_cores_averaged[:,5] = hpx_cores_averaged[:,5] / 1000000.0
for i in range (6,10):
    hpx_cores_averaged[:,i] = hpx_cores_averaged[:,i] / (1000000.0 * hpx_cores_averaged[:,0])

# read cores file single precision
hpx_cores_sp_matrix = np.genfromtxt(os.path.abspath('./data_hpx/cores_hpx_right_200_sp.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_hpx_data = int(hpx_cores_matrix.shape[0]/n_loop_hpx)
hpx_cores_sp_averaged = np.zeros((n_entries_hpx_data, hpx_cores_sp_matrix.shape[1] - 1))
for i in range (n_entries_hpx_data):
    hpx_cores_sp_averaged[i,:] = np.mean(hpx_cores_sp_matrix[i*n_loop_hpx:(i+1)*n_loop_hpx,1:],axis=0)
hpx_cores_sp_averaged[:,5] = hpx_cores_sp_averaged[:,5] / 1000000.0
for i in range (6,10):
    hpx_cores_sp_averaged[:,i] = hpx_cores_sp_averaged[:,i] / (1000000.0 * hpx_cores_sp_averaged[:,0])

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

# read blas file
hpx_blas_matrix = np.genfromtxt(os.path.abspath('./data_hpx/blas_hpx.txt'), dtype='float', delimiter=';' , skip_header=1)

################################################################################
# HPX TILE SCALING
# plot HPX Runtime distribution tiles_right
points = hpx_tiles_right_128_averaged[:,0]
plt.figure(figsize=(10,5))
plt.plot(points, hpx_tiles_right_18_cpu_averaged[:,4], 's--', c=colors[2], linewidth=1, label='18 Cores')
plt.plot(points, hpx_tiles_right_18_gpu_averaged[:,4], 's--', c=colors[4], linewidth=1, label='18 Cores + GPU')
plt.plot(points, hpx_tiles_right_16_averaged[:,4], 'o-', c=colors[1], linewidth=1, label='16 Cores')
plt.plot(points, hpx_tiles_right_128_averaged[:,4], 'o-', c=colors[0], linewidth=1, label='128 Cores')
#plt.title('Tile scaling HPX implementation for different tile sizes')
plt.legend(loc='lower left')
plt.xlabel('Tile size and tiles per dimension ')
plt.xscale("log")
labels_x = (20000 / points).astype(int).astype(str)
for i in range(0,labels_x.size):
    labels_x[i] = labels_x[i] + "\n T = " + points[i].astype(int).astype(str)
plt.xticks(ticks=points, labels=labels_x)
plt.yscale("log")
plt.ylabel('Time in s')
plt.savefig('figures/tiles_right_scaling_hpx.pdf', bbox_inches='tight')

################################################################################
# CORES RUNTIME DISTRIBUTION
# PETSc data
points = petsc_cores_averaged[:,0]
prediction = petsc_cores_averaged[:,7]
solve = petsc_cores_averaged[:,6]
assembly = petsc_cores_averaged[:,5]
plt.figure(figsize=(10,5))
plt.plot(points, petsc_cores_averaged[:,4], 's--', c=colors[0], linewidth=1, label='PETSc Total')
plt.plot(points, solve, 's--', c=colors[4], linewidth=1, label='PETSc Cholesky')
plt.plot(points, assembly, 's--', c=colors[2], linewidth=1, label='PETSc Assembly')
plt.plot(points, prediction, 's--', c=colors[1], linewidth=1, label='PETSc Prediction')
# HPX data
points = hpx_cores_averaged[:,0]
prediction = hpx_cores_averaged[:,9]
solve = hpx_cores_averaged[:,8]
choleksy = hpx_cores_averaged[:,7] + solve
assembly = hpx_cores_averaged[:,6]
total = hpx_cores_averaged[:,5]
plt.plot(points, total, 'o-', c=colors[0], linewidth=1, label='HPX Total')
plt.plot(points, choleksy, 'o-', c=colors[4], linewidth=1, label='HPX Cholesky')
plt.plot(points, assembly, 'o-', c=colors[2], linewidth=1, label='HPX Assembly')
plt.plot(points, prediction, 'o-', c=colors[1], linewidth=1, label='HPX Prediction')
#plt.title('Runtime distribution of HPX and PETSc for different number of cores')
plt.legend(loc='lower left')
plt.xlabel('N cores')
plt.xscale("log")
labels_x = points.astype(int).astype(str)
plt.xticks(ticks=points, labels= labels_x)
plt.yscale("log")
plt.ylabel('Time in s')
plt.savefig('figures/cores_distribution.pdf', format='pdf', bbox_inches='tight')

################################################################################
# CORES PARALLEL EFFICIENCY
# plot PETSc and HPX data scaling parallel efficiency
plt.figure(figsize=(10,5))
# line
plt.plot(points, 100 *np.ones(points.size), 'k:', linewidth=1)
# PETSc data
points = petsc_cores_averaged[:,0]
parallel_efficieny = 100 * petsc_cores_averaged[0,4] / (petsc_cores_averaged[:,4] * petsc_cores_averaged[:,0])
plt.plot(points, parallel_efficieny, 's--', c=greyscale[0], linewidth=1, label='PETSc FP64')
# HPX data
points = hpx_cores_averaged[:,0]
parallel_efficieny = 100 * hpx_cores_averaged[0,5] / (hpx_cores_averaged[:,5] * hpx_cores_averaged[:,0])
plt.plot(points, parallel_efficieny, 'o-', c=greyscale[2], linewidth=1, label='HPX FP64')
points = hpx_cores_sp_averaged[:,0]
parallel_efficieny = 100 * hpx_cores_sp_averaged[0,5] / (hpx_cores_sp_averaged[:,5] * hpx_cores_sp_averaged[:,0])
plt.plot(points, parallel_efficieny, 'o-', c=greyscale[4], linewidth=1, label='HPX FP32')
#plt.title('Parallel efficiency of HPX and PETSc for different number of cores')
plt.legend(loc='lower left')
plt.xlabel('N cores')
plt.xscale("log")
labels_x = points.astype(int).astype(str)
plt.xticks(ticks=points, labels= labels_x)
plt.ylabel('Parallel efficiency in %')
ticks_y = np.linspace(50, 100, num=6, endpoint=True, dtype=int)
plt.yticks(ticks=ticks_y)
plt.savefig('figures/cores_efficiency.pdf', bbox_inches='tight')

################################################################################
# CORES PARALLEL SPEEDUP
# plot PETSc and HPX data scaling parallel efficiency
plt.figure(figsize=(10,5))
# line
plt.plot(points,points, 'k:', linewidth=1)
# PETSc data
points = petsc_cores_averaged[:,0]
parallel_speedup = petsc_cores_averaged[0,4] / petsc_cores_averaged[:,4]
plt.plot(points, parallel_speedup, 's--', c=greyscale[0], linewidth=1, label='PETSc')
# HPX data
points = hpx_cores_averaged[:,0]
parallel_speedup = hpx_cores_averaged[0,5] / hpx_cores_averaged[:,5]
plt.plot(points, parallel_speedup, 'o-', c=greyscale[3], linewidth=1, label='HPX FP64')
points = hpx_cores_sp_averaged[:,0]
parallel_speedup = hpx_cores_sp_averaged[0,5] / hpx_cores_sp_averaged[:,5]
plt.plot(points, parallel_speedup, 'o-', c=greyscale[4], linewidth=1, label='HPX FP32')
#plt.title('Parallel speedup of HPX and PETSc for different number of cores')
plt.legend(loc='upper left')
plt.xlabel('N cores')
plt.xscale("log")
labels_x = points.astype(int).astype(str)
plt.xticks(ticks=points, labels= labels_x)
plt.ylabel('Parallel speedup')
plt.yscale("log")
ticks_y = np.logspace(0, 2, num=3, endpoint=True, base=10.0, dtype=int)
plt.yticks(ticks=ticks_y, labels=ticks_y)
plt.savefig('figures/cores_speedup.pdf', bbox_inches='tight')

################################################################################
# BLAS COMPARISON
# plot PETSc and HPX blas scaling
plt.figure(figsize=(10,5))
plt.plot(petsc_blas_matrix[:,0], petsc_blas_matrix[:,1], 's--', c=colors[4], linewidth=1, label='POTRF PETSc with fblaslapack')
plt.plot(petsc_blas_matrix[:,0], petsc_blas_matrix[:,2], 's--', c=colors[2], linewidth=1, label='TRSM PETSc with fblaslapack')
plt.plot(petsc_blas_matrix[:,0], petsc_blas_matrix[:,3], 's--', c=colors[1], linewidth=1, label='GEMM PETSc with fblaslapack')
plt.plot(hpx_blas_matrix[:,0], hpx_blas_matrix[:,1], 'o-', c=colors[4], linewidth=1, label='POTRF uBLAS')
plt.plot(hpx_blas_matrix[:,0], hpx_blas_matrix[:,2], 'o-', c=colors[2], linewidth=1, label='TRSM uBLAS')
plt.plot(hpx_blas_matrix[:,0], hpx_blas_matrix[:,3], 'o-', c=colors[1], linewidth=1, label='GEMM uBLAS')
#plt.title('Comparison of uBLAS and PETSc with fblaslapack for different training set sizes')
plt.legend()
plt.xlabel('N matrix dimension')
plt.xticks(petsc_blas_matrix[:,0])
plt.ylabel('Time in s')
plt.xscale("log")
plt.yscale("log")
plt.savefig('figures/blas_hpx_petsc_comparison.pdf', bbox_inches='tight')

print('All figures generated.')
