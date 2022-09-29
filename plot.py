import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
################################################################################
# READ PETSC FILES
# get header and loop number for averaging
petsc_header = np.genfromtxt(os.path.abspath('./data_petsc/data_petsc.txt'), dtype='unicode', delimiter=';' , max_rows=1)
n_loop_petsc = int(petsc_header[-1])
# read cores file
petsc_cores_matrix = np.genfromtxt(os.path.abspath('./data_petsc/cores_petsc.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_petsc_cores = int(petsc_cores_matrix.shape[0]/n_loop_petsc)
petsc_cores_averaged = np.zeros((n_entries_petsc_cores, petsc_cores_matrix.shape[1] - 1))
for i in range (n_entries_petsc_cores):
    petsc_cores_averaged[i,:] = np.mean(petsc_cores_matrix[i*n_loop_petsc:(i+1)*n_loop_petsc,:-1],axis=0)
# read data file
petsc_data_matrix = np.genfromtxt(os.path.abspath('./data_petsc/data_petsc.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_petsc_data = int(petsc_data_matrix.shape[0]/n_loop_petsc)
petsc_data_averaged = np.zeros((n_entries_petsc_data, petsc_data_matrix.shape[1] - 1))
for i in range (n_entries_petsc_data):
    petsc_data_averaged[i,:] = np.mean(petsc_data_matrix[i*n_loop_petsc:(i+1)*n_loop_petsc,:-1],axis=0)

# read blas file
petsc_blas_matrix = np.genfromtxt(os.path.abspath('./data_petsc/data_petsc_blas.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_petsc_blas = int(petsc_blas_matrix.shape[0]/n_loop_petsc)
petsc_blas_averaged = np.zeros((n_entries_petsc_blas, petsc_blas_matrix.shape[1] - 1))
for i in range (n_entries_petsc_blas):
    petsc_blas_averaged[i,:] = np.mean(petsc_blas_matrix[i*n_loop_petsc:(i+1)*n_loop_petsc,:-1],axis=0)
################################################################################
# READ HPX FILES
# get header and loop number for averaging
hpx_header = np.genfromtxt(os.path.abspath('./data_hpx/data_hpx_left_10.txt'), dtype='unicode', delimiter=';' , max_rows=1)
n_loop_hpx = int(hpx_header[-1])
#######################
# LEFT LOOKING CHOLESKY
# read tiles file
hpx_tiles_left_matrix = np.genfromtxt(os.path.abspath('./data_hpx/tiles_hpx_left.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_hpx_tiles = int(hpx_tiles_left_matrix.shape[0]/n_loop_hpx)
n_cores_hpx = hpx_tiles_left_matrix[0,1]
hpx_tiles_left_averaged = np.zeros((n_entries_hpx_tiles, hpx_tiles_left_matrix.shape[1] - 2))
for i in range (n_entries_hpx_tiles):
    hpx_tiles_left_averaged[i,:] = np.mean(hpx_tiles_left_matrix[i*n_loop_petsc:(i+1)*n_loop_petsc,2:],axis=0)
hpx_tiles_left_averaged[:,4] = hpx_tiles_left_averaged[:,4] / 1000000.0
for i in range(n_entries_hpx_tiles):
    n_tiles = hpx_tiles_left_averaged[i,0] * hpx_tiles_left_averaged[i,0]
    n_cores = n_cores_hpx
    if (n_tiles < n_cores):
        divider = n_tiles
    else:
        divider = n_cores
    #divider = 1
    hpx_tiles_left_averaged[i,5:9] = hpx_tiles_left_averaged[i,5:9] / (1000000.0 * divider)

# read data file
hpx_data_left_10_matrix = np.genfromtxt(os.path.abspath('./data_hpx/data_hpx_left_10.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_hpx_data = int(hpx_data_left_10_matrix.shape[0]/n_loop_hpx)
hpx_data_left_10_averaged = np.zeros((n_entries_hpx_data, hpx_data_left_10_matrix.shape[1] - 2))
for i in range (n_entries_hpx_data):
    hpx_data_left_10_averaged[i,:] = np.mean(hpx_data_left_10_matrix[i*n_loop_petsc:(i+1)*n_loop_petsc,2:],axis=0)
hpx_data_left_10_averaged[:,4] = hpx_data_left_10_averaged[:,4] / 1000000.0
hpx_data_left_10_averaged[:,5:9] = hpx_data_left_10_averaged[:,5:9] / (1000000.0 * n_cores_hpx)

# read data file for 100 tiles
hpx_data_left_100_matrix = np.genfromtxt(os.path.abspath('./data_hpx/data_hpx_left_100.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_hpx_data = int(hpx_data_left_100_matrix.shape[0]/n_loop_hpx)
hpx_data_left_100_averaged = np.zeros((n_entries_hpx_data, hpx_data_left_100_matrix.shape[1] - 2))
for i in range (n_entries_hpx_data):
    hpx_data_left_100_averaged[i,:] = np.mean(hpx_data_left_100_matrix[i*n_loop_petsc:(i+1)*n_loop_petsc,2:],axis=0)
hpx_data_left_100_averaged[:,4] = hpx_data_left_100_averaged[:,4] / 1000000.0
hpx_data_left_100_averaged[:,5:9] = hpx_data_left_100_averaged[:,5:9] / (1000000.0 * n_cores_hpx)

# read data file for 200 tiles
hpx_data_left_200_matrix = np.genfromtxt(os.path.abspath('./data_hpx/data_hpx_left_200.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_hpx_data = int(hpx_data_left_200_matrix.shape[0]/n_loop_hpx)
hpx_data_left_200_averaged = np.zeros((n_entries_hpx_data, hpx_data_left_200_matrix.shape[1] - 2))
for i in range (n_entries_hpx_data):
    hpx_data_left_200_averaged[i,:] = np.mean(hpx_data_left_200_matrix[i*n_loop_petsc:(i+1)*n_loop_petsc,2:],axis=0)
hpx_data_left_200_averaged[:,4] = hpx_data_left_200_averaged[:,4] / 1000000.0
hpx_data_left_200_averaged[:,5:9] = hpx_data_left_200_averaged[:,5:9] / (1000000.0 * n_cores_hpx)
# ######################## REWORK SCALING
# RIGHT LOOKING CHOLESKY
# read tiles file
hpx_tiles_right_matrix = np.genfromtxt(os.path.abspath('./data_hpx/tiles_hpx_right.txt'), dtype='float', delimiter=';' , skip_header=1)
hpx_tiles_right_averaged = np.zeros((n_entries_hpx_tiles, hpx_tiles_right_matrix.shape[1] - 2))
for i in range (n_entries_hpx_tiles):
    hpx_tiles_right_averaged[i,:] = np.mean(hpx_tiles_right_matrix[i*n_loop_petsc:(i+1)*n_loop_petsc,2:],axis=0)
hpx_tiles_right_averaged[:,4] = hpx_tiles_right_averaged[:,4] / 1000000.0
for i in range(n_entries_hpx_tiles):
    n_tiles = hpx_tiles_right_averaged[i,0] * hpx_tiles_right_averaged[i,0]
    n_cores = n_cores_hpx
    if (n_tiles < n_cores):
        divider = n_tiles
    else:
        divider = n_cores
    #divider = 1
    hpx_tiles_right_averaged[i,5:9] = hpx_tiles_right_averaged[i,5:9] / (1000000.0 * divider)
# read data file
hpx_data_right_matrix = np.genfromtxt(os.path.abspath('./data_hpx/data_hpx_right.txt'), dtype='float', delimiter=';' , skip_header=1)
hpx_data_right_averaged = np.zeros((n_entries_hpx_data, hpx_data_right_matrix.shape[1] - 2))
for i in range (n_entries_hpx_data):
    hpx_data_right_averaged[i,:] = np.mean(hpx_data_right_matrix[i*n_loop_petsc:(i+1)*n_loop_petsc,2:],axis=0)
hpx_data_right_averaged[:,4] = hpx_data_right_averaged[:,4] / 1000000.0
hpx_data_right_averaged[:,5:9] = hpx_data_right_averaged[:,5:9] / (1000000.0 * n_cores_hpx)
######################
# TOP LOOKING CHOLESKY
# read tiles file
hpx_tiles_top_matrix = np.genfromtxt(os.path.abspath('./data_hpx/tiles_hpx_top.txt'), dtype='float', delimiter=';' , skip_header=1)
hpx_tiles_top_averaged = np.zeros((n_entries_hpx_tiles, hpx_tiles_top_matrix.shape[1] - 2))
for i in range (n_entries_hpx_tiles):
    hpx_tiles_top_averaged[i,:] = np.mean(hpx_tiles_top_matrix[i*n_loop_petsc:(i+1)*n_loop_petsc,2:],axis=0)
hpx_tiles_top_averaged[:,4] = hpx_tiles_top_averaged[:,4] / 1000000.0
for i in range(n_entries_hpx_tiles):
    n_tiles = hpx_tiles_top_averaged[i,0] * hpx_tiles_top_averaged[i,0]
    n_cores = n_cores_hpx
    if (n_tiles < n_cores):
        divider = n_tiles
    else:
        divider = n_cores
    #divider = 1
    hpx_tiles_top_averaged[i,5:9] = hpx_tiles_top_averaged[i,5:9] / (1000000.0 * divider)
# read data file
hpx_data_top_matrix = np.genfromtxt(os.path.abspath('./data_hpx/data_hpx_top.txt'), dtype='float', delimiter=';' , skip_header=1)
hpx_data_top_averaged = np.zeros((n_entries_hpx_data, hpx_data_top_matrix.shape[1] - 2))
for i in range (n_entries_hpx_data):
    hpx_data_top_averaged[i,:] = np.mean(hpx_data_top_matrix[i*n_loop_petsc:(i+1)*n_loop_petsc,2:],axis=0)
hpx_data_top_averaged[:,4] = hpx_data_top_averaged[:,4] / 1000000.0
hpx_data_top_averaged[:,5:9] = hpx_data_top_averaged[:,5:9] / (1000000.0 * n_cores_hpx)

######################
# BLAS
# read blas file
hpx_blas_matrix = np.genfromtxt(os.path.abspath('./data_hpx/data_hpx_blas.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_hpx_blas = int(hpx_blas_matrix.shape[0]/n_loop_hpx)
hpx_blas_averaged = np.zeros((n_entries_hpx_blas, hpx_blas_matrix.shape[1] - 2))
for i in range (n_entries_hpx_blas):
    hpx_blas_averaged[i,:] = np.mean(hpx_blas_matrix[i*n_loop_petsc:(i+1)*n_loop_petsc,2:],axis=0)
hpx_blas_averaged[:,6] = hpx_blas_averaged[:,6] / 1000000.0
hpx_blas_averaged[:,7] = hpx_blas_averaged[:,7] / 1000000.0
################################################################################
# PLOTS
# plot error from PETSc data (identical to HPX)
plt.figure(figsize=(6,4))
plt.plot(petsc_data_averaged[:,1], petsc_data_averaged[:,-1], 'ko-', label='Error', linewidth=2)
plt.title('Test error for different training set sizes')
plt.xlabel('N training samples')
plt.xticks(petsc_data_averaged[:,1])
plt.ylabel('Error')
plt.xscale("log")
plt.yscale("log")
plt.savefig('figures/error_petsc.pdf')
################################################################################
# DIFFERENT TILED ALGORITHMS
# # plot HPX tile scaling for different tiled choleksy algorithms
# plt.figure(figsize=(6,4))
# plt.plot(hpx_tiles_left_averaged[:,0], hpx_tiles_left_averaged[:,6], 'go-', label='left-looking', linewidth=2)
# plt.plot(hpx_tiles_right_averaged[:,0], hpx_tiles_right_averaged[:,6], 'bo-', label='right-looking', linewidth=2)
# plt.plot(hpx_tiles_top_averaged[:,0], hpx_tiles_top_averaged[:,6], 'ro-', label='top-looking', linewidth=2)
# plt.title('Runtime of different tiled Cholesky decompositions for different numbers of tiles')
# plt.legend()
# plt.xlabel('N tiles')
# plt.xticks(hpx_tiles_left_averaged[:,0])
# plt.ylabel('Time in s')
# plt.xscale("log")
# plt.yscale("log")
# plt.savefig('figures/tiles_cholesky_hpx.pdf')

# # plot HPX data scaling for different tiled choleksy algorithms
# plt.figure(figsize=(6,4))
# plt.plot(hpx_data_left_averaged[:,1], hpx_data_left_averaged[:,6] + hpx_data_left_averaged[:,7], 'go-', label='left-looking', linewidth=2)
# # plt.plot(hpx_data_right_averaged[:,1], hpx_data_right_averaged[:,6] + hpx_data_right_averaged[:,7], 'bo-', label='right-looking', linewidth=2)
# # plt.plot(hpx_data_top_averaged[:,1], hpx_data_top_averaged[:,6] + hpx_data_top_averaged[:,7], 'ro-', label='top-looking', linewidth=2)
# plt.title('Runtime of different tiled Cholesky decompositions for different training set sizes')
# plt.legend()
# plt.xlabel('N training samples')
# plt.xticks(hpx_data_left_averaged[:,1])
# plt.ylabel('Time in s')
# plt.xscale("log")
# plt.yscale("log")
# plt.savefig('figures/data_cholesky_hpx.pdf')
################################################################################
# DATA SCALING
# plot PETSc and HPX data scaling for different tiled choleksy algorithms
plt.figure(figsize=(6,4))
plt.plot(petsc_data_averaged[:,1], petsc_data_averaged[:,6], 'ko-', label='PETSc', linewidth=2)
plt.plot(hpx_data_left_10_averaged[:,1], hpx_data_left_10_averaged[:,6] + hpx_data_left_10_averaged[:,7], 'go-', label='HPX 10', linewidth=2)
plt.plot(hpx_data_left_100_averaged[:,1], hpx_data_left_100_averaged[:,6] + hpx_data_left_100_averaged[:,7], 'bo-', label='HPX 100', linewidth=2)
plt.plot(hpx_data_left_200_averaged[:,1], hpx_data_left_200_averaged[:,6] + hpx_data_left_200_averaged[:,7], 'ro-', label='HPX 200', linewidth=2)
plt.title('Choleskly solve runtime of PETSc and HPX for different training set sizes')
plt.legend()
plt.xlabel('N training samples')
plt.xticks(petsc_data_averaged[:,1])
plt.ylabel('Time in s')
plt.xscale("log")
plt.yscale("log")
plt.savefig('figures/data_cholesky_hpx_petsc_comparison.pdf')

# plot PETSc and HPX data scaling for total time
plt.figure(figsize=(6,4))
plt.plot(petsc_data_averaged[:,1], petsc_data_averaged[:,4], 'ko-', label='PETSc', linewidth=2)
plt.plot(hpx_data_left_10_averaged[:,1], hpx_data_left_10_averaged[:,4], 'go-', label='HPX 10', linewidth=2)
plt.plot(hpx_data_left_100_averaged[:,1], hpx_data_left_100_averaged[:,4], 'bo-', label='HPX 100', linewidth=2)
plt.plot(hpx_data_left_200_averaged[:,1], hpx_data_left_200_averaged[:,4], 'ro-', label='HPX 200', linewidth=2)
plt.title('Total Runtime of PETSc and HPX for different training set sizes')
plt.legend()
plt.xlabel('N training samples')
plt.xticks(petsc_data_averaged[:,1])
plt.ylabel('Time in s')
plt.xscale("log")
plt.yscale("log")
plt.savefig('figures/data_total_hpx_petsc_comparison.pdf')
################################################################################
# HPX TILE RUNTIME DISTRIBUTION
# plot HPX Runtime distribution tiles_left
points = hpx_tiles_left_averaged[:,0]
prediction = hpx_tiles_left_averaged[:,8]
solve = hpx_tiles_left_averaged[:,7]
choleksy = hpx_tiles_left_averaged[:,6]
assembly = hpx_tiles_left_averaged[:,5]
plt.figure(figsize=(6,4))
plt.plot(points, hpx_tiles_left_averaged[:,4], 'ko-', label='Total Runtime', linewidth=2)
plt.plot(points, assembly, 'bo-', label='Assembly')
plt.plot(points, choleksy, 'go-', label='Cholesky')
plt.plot(points, solve, 'yo-', label='Triangular Solve')
plt.plot(points, prediction, 'ro-', label='Prediction')
plt.title('Runtime distribution HPX for different number of tiles')
plt.legend()
plt.xlabel('N tiles')
plt.xticks(points)
plt.xscale("log")
plt.yscale("log")
plt.ylabel('Time in s')
plt.savefig('figures/tiles_left_distribution_hpx.pdf')
# # plot HPX Runtime distribution data size
# points = hpx_data_left_averaged[:,1]
# prediction = hpx_data_left_averaged[:,8]
# solve = hpx_data_left_averaged[:,7]
# choleksy = hpx_data_left_averaged[:,6]
# assembly = hpx_data_left_averaged[:,5]
# plt.figure(figsize=(6,4))
# plt.plot(points, hpx_data_left_averaged[:,4], 'ko-', label='Total Runtime', linewidth=2)
# plt.plot(points, assembly, 'bo-', label='Assembly')
# plt.plot(points, choleksy, 'go-', label='Cholesky Solve')
# plt.plot(points, solve, 'yo-', label='Triangular Solve')
# plt.plot(points, prediction, 'ro-', label='Prediction')
# plt.title('Runtime distribution HPX for different training set sizes')
# plt.legend()
# plt.xlabel('N training samples')
# plt.xticks(points)
# plt.xscale("log")
# plt.yscale("log")
# plt.ylabel('Time in s')
# plt.savefig('figures/data_left_distribution_hpx.pdf')
################################################################################
# CORES RUNTIME DISTRIBUTION
# PETSc data
points = petsc_cores_averaged[:,0]
prediction = petsc_cores_averaged[:,7]
solve = petsc_cores_averaged[:,6]
assembly = petsc_cores_averaged[:,5]
plt.figure(figsize=(6,8))
plt.plot(points, petsc_cores_averaged[:,4], 'ks-', label='Total Runtime', linewidth=2)
plt.plot(points, assembly, 'bs-', label='Assembly')
plt.plot(points, solve, 'gs-', label='Cholesky Solve')
plt.plot(points, prediction, 'rs-', label='Prediction')
# HPX data
points = hpx_tiles_left_averaged[:,0]
prediction = hpx_tiles_left_averaged[:,8]
solve = hpx_tiles_left_averaged[:,7]
choleksy = hpx_tiles_left_averaged[:,6] + solve
assembly = hpx_tiles_left_averaged[:,5]
plt.plot(points, hpx_tiles_left_averaged[:,4], 'ko-', label='Total Runtime', linewidth=2)
plt.plot(points, assembly, 'bo-', label='Assembly')
plt.plot(points, choleksy, 'go-', label='Cholesky Solve')
plt.plot(points, prediction, 'ro-', label='Prediction')
plt.title('Runtime distribution of HPX and PETSc for different number of cores')
plt.legend()
plt.xlabel('N cores')
plt.xticks(points)
plt.xscale("log")
plt.yscale("log")
plt.ylabel('Time in s')
plt.savefig('figures/cores_distribution.pdf')
################################################################################
# DATA RUNTIME DISTRIBUTION
# PETSc data
points = petsc_data_averaged[:,1]
prediction = petsc_data_averaged[:,7]
solve = petsc_data_averaged[:,6]
assembly = petsc_data_averaged[:,5]
plt.figure(figsize=(6,8))
plt.plot(points, petsc_data_averaged[:,4], 'ks-', label='Total Runtime', linewidth=2)
plt.plot(points, assembly, 'bs-', label='Assembly')
plt.plot(points, solve, 'gs-', label='Cholesky Solve')
plt.plot(points, prediction, 'rs-', label='Prediction')
# HPX data
points = np.hstack((hpx_data_left_100_averaged[:,1],hpx_data_left_200_averaged[:,1]))
prediction = np.hstack((hpx_data_left_100_averaged[:,8],hpx_data_left_200_averaged[:,8]))
solve = np.hstack((hpx_data_left_100_averaged[:,7],hpx_data_left_200_averaged[:,7]))
choleksy = np.hstack((hpx_data_left_100_averaged[:,6],hpx_data_left_200_averaged[:,6])) + solve
assembly = np.hstack((hpx_data_left_100_averaged[:,5],hpx_data_left_200_averaged[:,5]))
total = np.hstack((hpx_data_left_100_averaged[:,4],hpx_data_left_200_averaged[:,4]))
plt.plot(points, total, 'ko-', label='Total Runtime', linewidth=2)
plt.plot(points, assembly, 'bo-', label='Assembly')
plt.plot(points, choleksy, 'go-', label='Cholesky Solve')
plt.plot(points, prediction, 'ro-', label='Prediction')
plt.title('Runtime distribution of PETSc and HPX for different training set sizes')
plt.legend()
plt.xlabel('N training samples')
plt.xticks(points)
plt.xscale("log")
plt.yscale("log")
plt.ylabel('Time in s')
plt.savefig('figures/data_distribution.pdf')
################################################################################
# BLAS COMPARISON
# plot PETSc and HPX blas scaling
plt.figure(figsize=(6,4))
plt.plot(petsc_blas_averaged[:,1], petsc_blas_averaged[:,6], 'ko-', label='PETSc fblaslapack', linewidth=2)
plt.plot(hpx_blas_averaged[:,1], hpx_blas_averaged[:,6] + hpx_blas_averaged[:,7], 'go-', label='HPX uBLAS', linewidth=2)
plt.title('Choleskly solve runtime of PETSc and HPX BLAS libraries for different training set sizes')
plt.legend()
plt.xlabel('N training samples')
plt.xticks(petsc_blas_averaged[:,1])
plt.ylabel('Time in s')
plt.xscale("log")
plt.yscale("log")
plt.savefig('figures/blas_cholesky_hpx_petsc_comparison.pdf')


print('All figures generated.')
