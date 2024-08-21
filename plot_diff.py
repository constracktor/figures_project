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

# read cores file fblaslapack
petsc_cores_matrix = np.genfromtxt(os.path.abspath('./data_petsc/cores_petsc.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_petsc_cores = int(petsc_cores_matrix.shape[0]/n_loop_petsc)
petsc_cores_averaged = np.zeros((n_entries_petsc_cores, petsc_cores_matrix.shape[1] - 1))
for i in range (n_entries_petsc_cores):
    petsc_cores_averaged[i,:] = np.mean(petsc_cores_matrix[i*n_loop_petsc:(i+1)*n_loop_petsc,:-1],axis=0)


n_loop_petsc = 1
# read cores file mkl
petsc_cores_mkl_averaged = np.genfromtxt(os.path.abspath('./data_petsc/cores_petsc_mkl.txt'), dtype='float', delimiter=';' , skip_header=1)

################################################################################
# READ HPX FILES
# get header and loop number for averaging
hpx_header = np.genfromtxt(os.path.abspath('./data_hpx/cores_hpx_right_200.txt'), dtype='unicode', delimiter=';' , max_rows=1)
n_loop_hpx = int(hpx_header[-1])

# read cores file double precision ublas
hpx_cores_matrix = np.genfromtxt(os.path.abspath('./data_hpx/cores_hpx_right_200_sp.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_hpx_data = int(hpx_cores_matrix.shape[0]/n_loop_hpx)
hpx_cores_averaged = np.zeros((n_entries_hpx_data, hpx_cores_matrix.shape[1] - 1))
for i in range (n_entries_hpx_data):
    hpx_cores_averaged[i,:] = np.mean(hpx_cores_matrix[i*n_loop_hpx:(i+1)*n_loop_hpx,1:],axis=0)
hpx_cores_averaged[:,5] = hpx_cores_averaged[:,5] / 1000000.0
for i in range (6,10):
    hpx_cores_averaged[:,i] = hpx_cores_averaged[:,i] / (1000000.0 * hpx_cores_averaged[:,0])


n_loop_hpx = 1
# read cores file double precision ublas
hpx_cores_matrix_mkl = np.genfromtxt(os.path.abspath('./data_hpx/cores_hpx_right_200_mkl.txt'), dtype='float', delimiter=';' , skip_header=1)
n_entries_hpx_data = int(hpx_cores_matrix_mkl.shape[0]/n_loop_hpx)
hpx_cores_averaged_mkl = np.zeros((n_entries_hpx_data, hpx_cores_matrix_mkl.shape[1] - 1))
for i in range (n_entries_hpx_data):
    hpx_cores_averaged_mkl[i,:] = np.mean(hpx_cores_matrix_mkl[i*n_loop_hpx:(i+1)*n_loop_hpx,1:],axis=0)
hpx_cores_averaged_mkl[:,5] = hpx_cores_averaged_mkl[:,5] / 1000000.0
for i in range (6,10):
    hpx_cores_averaged_mkl[:,i] = hpx_cores_averaged_mkl[:,i] / (1000000.0 * hpx_cores_averaged_mkl[:,0])



################################################################################
# CORES RUNTIME DISTRIBUTION OLD
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
plt.ylabel('Runtime in s')
plt.savefig('figures/cores_distribution_old.pdf', format='pdf', bbox_inches='tight')
################################################################################
# CORES RUNTIME DISTRIBUTION NEW
# PETSc data
points = petsc_cores_mkl_averaged[:,0]
prediction = petsc_cores_mkl_averaged[:,7]
solve = petsc_cores_mkl_averaged[:,6]
assembly = petsc_cores_mkl_averaged[:,5]
plt.figure(figsize=(10,5))
plt.plot(points, petsc_cores_mkl_averaged[:,4], 's--', c=colors[0], linewidth=1, label='PETSc Total')
plt.plot(points, solve, 's--', c=colors[4], linewidth=1, label='PETSc Cholesky')
plt.plot(points, assembly, 's--', c=colors[2], linewidth=1, label='PETSc Assembly')
plt.plot(points, prediction, 's--', c=colors[1], linewidth=1, label='PETSc Prediction')
# HPX data
points = hpx_cores_averaged_mkl[:,0]
prediction = hpx_cores_averaged_mkl[:,9]
solve = hpx_cores_averaged_mkl[:,8]
choleksy = hpx_cores_averaged_mkl[:,7] + solve
assembly = hpx_cores_averaged_mkl[:,6]
total = hpx_cores_averaged_mkl[:,5]
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
plt.ylabel('Runtime in s')
plt.savefig('figures/cores_distribution_new.pdf', format='pdf', bbox_inches='tight')
################################################################################
# CORES RUNTIME DISTRIBUTION HPX
# HPX data old
points = hpx_cores_averaged[:,0]
solve = hpx_cores_averaged[:,8]
choleksy = hpx_cores_averaged[:,7] + solve
assembly = hpx_cores_averaged[:,6]
total = hpx_cores_averaged[:,5]
plt.figure(figsize=(10,5))
plt.plot(points, total, 'o--', c=colors[0], linewidth=1, label='HPX Total ublas')
plt.plot(points, choleksy, 'o--', c=colors[4], linewidth=1, label='HPX Cholesky ublas')
plt.plot(points, assembly, 'o--', c=colors[2], linewidth=1, label='HPX Assembly ublas')
# HPX data new
points = hpx_cores_averaged_mkl[:,0]
solve = hpx_cores_averaged_mkl[:,8]
choleksy_mkl = hpx_cores_averaged_mkl[:,7] + solve
assembly = hpx_cores_averaged_mkl[:,6]
total = hpx_cores_averaged_mkl[:,5]
plt.plot(points, total, 'o-', c=colors[0], linewidth=1, label='HPX Total mkl')
plt.plot(points, choleksy_mkl, 'o-', c=colors[4], linewidth=1, label='HPX Cholesky mkl')
plt.plot(points, assembly, 'o-', c=colors[2], linewidth=1, label='HPX Assembly mkl')
#plt.title('Runtime distribution of HPX and PETSc for different number of cores')
plt.legend(loc='lower left')
plt.xlabel('N cores')
plt.xscale("log")
#
factor = np.round(choleksy/choleksy_mkl,2)
labels_x = points.astype(int).astype(str)
for i in range(0,labels_x.size):
    labels_x[i] = labels_x[i] + "\ns=" + factor[i].astype(str)
plt.xticks(ticks=points, labels=labels_x)
plt.yscale("log")
plt.ylabel('Runtime in s')
plt.savefig('figures/cores_distribution_hpx.pdf', format='pdf', bbox_inches='tight')

################################################################################
# CORES RUNTIME DISTRIBUTION PETSc
# PETSc data
points = petsc_cores_averaged[:,0]
solve = petsc_cores_averaged[:,6]
assembly = petsc_cores_averaged[:,5]
plt.figure(figsize=(10,5))
plt.plot(points, petsc_cores_averaged[:,4], 's--', c=colors[0], linewidth=1, label='PETSc Total fblas')
plt.plot(points, solve, 's--', c=colors[4], linewidth=1, label='PETSc Cholesky fblas')
plt.plot(points, assembly, 's--', c=colors[2], linewidth=1, label='PETSc Assembly fblas')
# PETSc data
points = petsc_cores_mkl_averaged[:,0]
solve_mkl = petsc_cores_mkl_averaged[:,6]
assembly = petsc_cores_mkl_averaged[:,5]
plt.plot(points, petsc_cores_mkl_averaged[:,4], 's-', c=colors[0], linewidth=1, label='PETSc Total mkl')
plt.plot(points, solve_mkl, 's-', c=colors[4], linewidth=1, label='PETSc Cholesky mkl')
plt.plot(points, assembly, 's-', c=colors[2], linewidth=1, label='PETSc Assembly mkl')
#plt.title('Runtime distribution of HPX and PETSc for different number of cores')
plt.legend(loc='lower left')
plt.xlabel('N cores')
plt.xscale("log")
factor = np.round(solve/solve_mkl,2)
labels_x = points.astype(int).astype(str)
for i in range(0,labels_x.size):
    labels_x[i] = labels_x[i] + "\ns=" + factor[i].astype(str)
plt.xticks(ticks=points, labels=labels_x)
plt.yscale("log")
plt.ylabel('Runtime in s')
plt.savefig('figures/cores_distribution_petsc.pdf', format='pdf', bbox_inches='tight')