import numpy as np
import matplotlib.pyplot as plt
import math
import random

def stdLeastSquare(x,y,z):
    A_t=np.vstack((x,z,np.ones(z.shape)))
    #print(A_t)
    y=y[np.newaxis]
    y_t = y.T
    A_t_B = np.matmul(A_t,y_t)
    A_t_A = np.matmul(A_t,A_t.transpose())
    coefficients = np.matmul(np.linalg.inv(A_t_A),A_t_B)
    return coefficients

def covariance(x,y,z):
    x_mean = sum(x)/len(x)
    print(x_mean)
    y_mean = sum(y)/len(y)
    print(y_mean)
    z_mean = sum(z)/len(z)
    print(z_mean)
    x_dev = (x-x_mean)
    y_dev = (y-y_mean)
    z_dev = (z-z_mean)
    cxx = np.dot(x_dev,x_dev)/len(x)
    cyy = np.dot(y_dev,y_dev)/len(y)
    czz = np.dot(z_dev,z_dev)/len(z)
    cxy = np.dot(x_dev,y_dev)/len(x)
    cyx = cxy
    cxz = np.dot(x_dev,z_dev)/len(x)
    czx=cxz
    cyz = np.dot(y_dev,z_dev)/len(y)
    czy = cyz
    cov_mat = np.array([[cxx, cyx, czx], [cxy, cyy, czy],[cxz, cyz, czz]])
    return cov_mat

#def calMagDir(x,y,z,cov_mat):
def calMagDir(cov_mat):
    eig_v, eig_vec = np.linalg.eig(cov_mat)
    print("The eigen values are :")
    print(eig_v)
    print("The Eigen vectors are :")
    print(eig_vec)
    if min(eig_v) == eig_v[0]:
        return eig_v[0], eig_vec[:,0]
    if min(eig_v) == eig_v[1]:
        return eig_v[1], eig_vec[:,1]
    if min(eig_v) == eig_v[2]:
        return eig_v[2], eig_vec[:,2]


def totalLeastSquare(x,y,z):
    cov=covariance(x,y,z)
    mag, coefficients = calMagDir(cov)
    return mag, coefficients


def calc_y_sls(a,x,c,z,d):
    return ((a*x)+(c*z)+d)

def calc_y_tls(a,x,b,c,z,d):
    return -(a/b*x)-(c/b)*z+(d/b)

## RANSAC Function

def ransac_func(N,s,t,x,y,z):
    random_sample_list=np.empty(s,dtype="int")
    counter_max=0
    final_d=0
    final_vector=np.array([])

    for j in range(N):
        counter=0
        for i in range(s):
            random_sample_list[i]=random.randint(0,len(x)-1)
        x_sample=np.zeros(s)
        y_sample=np.zeros(s)
        z_sample=np.zeros(s)
        count=0
        for i in random_sample_list:
            x_sample[count]=x[i]
            y_sample[count]=y[i]
            z_sample[count]=z[i]
            count=count+1
        points_mat=np.vstack((x_sample,y_sample,z_sample))
        vector_1=points_mat[:,1]-points_mat[:,0]
        vector_2=points_mat[:,2]-points_mat[:,0]
        normal=np.cross(vector_1,vector_2)
        d=np.dot(normal,points_mat[:,0])
        for i in range(len(x)):
            point=np.array([x[i],y[i],z[i]])
            err=abs((np.dot(point,normal)-d)/math.sqrt(np.dot(normal,normal)))
            if err<=t:
                counter=counter+1
        if counter>counter_max:
            counter_max=counter
            final_vector=normal
            final_d=d
    
    print("the counter value at the end of ransac operation is : number of inliers")
    print(counter)
    return final_vector,d


data = np.loadtxt('/home/vignesh/Desktop/SEM2/Perception/Project1/pc1.csv', dtype='float', delimiter = ",")
print(data)

x_1=data[:,0]
y_1=data[:,1]
z_1=data[:,2]

## 2.1 (a) To find the covariance using user defined function for the given set of points (x,y,z)
cov = covariance(x_1,y_1,z_1)
print("The covariance Matrix is :")
print(cov)

## 2.1 (b) Calculating the magnitude and direction of surface normal using covariance matrix
d, eigen_vector = calMagDir(cov)
print("The magintude of surface normal is :")
print(d)
print("The direction of surface normal is :")
print(eigen_vector)

## 2.2 (a) Calculating Standard Least square and total least square comparison for pc1.csv and pc2.csv data

### Standard least square method to plot the surface for pc1.csv data
coefficients_sls = stdLeastSquare(x_1,y_1,z_1)
print("The best coefficients using lease square method is :")
print(coefficients_sls)
x_mesh_d1, z_mesh_d1 = np.meshgrid(x_1,z_1)
y_mesh_d1_sls=calc_y_sls(coefficients_sls[0][0],x_mesh_d1,coefficients_sls[1][0],z_mesh_d1,coefficients_sls[2][0])

### Total least square method for pc1.csv data
mag, coefficients_tls = totalLeastSquare(x_1,y_1,z_1)
print("The best coefficients using total least square method is :")
print(coefficients_tls)
y_mesh_d1_tls=calc_y_tls(coefficients_tls[0],x_mesh_d1,coefficients_tls[1],coefficients_tls[2],z_mesh_d1,mag)

### Ransac for pc1.csv
coefficients_ransac,d_ransac=ransac_func(100,6,0.1,x_1,y_1,z_1)
y_mesh_d1_ransac=calc_y_tls(coefficients_ransac[0],x_mesh_d1,coefficients_ransac[1],coefficients_ransac[2],z_mesh_d1,d_ransac)

### Reading data from pc2.csv
data_2 = np.loadtxt('/home/vignesh/Desktop/SEM2/Perception/Project1/pc1.csv', dtype='float', delimiter = ",")
x_2=data_2[:,0]
y_2=data_2[:,1]
z_2=data_2[:,2]

### Standard least square method to plot the surface for pc2.csv data
coefficients_2_sls = stdLeastSquare(x_2,y_2,z_2)
print("The best coefficients using lease square method is :")
print(coefficients_2_sls)
x_mesh_d2, z_mesh_d2 = np.meshgrid(x_2,z_2)
y_mesh_d2_sls=calc_y_sls(coefficients_2_sls[0][0],x_mesh_d2,coefficients_2_sls[1][0],z_mesh_d2,coefficients_2_sls[2][0])

### Total least square method for pc2.csv data
mag_2, coefficients_2_tls = totalLeastSquare(x_2,y_2,z_2)
print("The best coefficients using total least square method is :")
print(coefficients_2_tls)
y_mesh_d2_tls=calc_y_tls(coefficients_2_tls[0],x_mesh_d2,coefficients_2_tls[1],coefficients_2_tls[2],z_mesh_d2,mag_2)

### Ransac for pc2.csv
coefficients_2_ransac,d_2_ransac=ransac_func(200,3,0.1,x_2,y_2,z_2)
y_mesh_d2_ransac=calc_y_tls(coefficients_2_ransac[0],x_mesh_d2,coefficients_2_ransac[1],coefficients_2_ransac[2],z_mesh_d2,d_2_ransac)

### Plotting the x,y,z in plane from data set pc1.csv
fig=plt.figure()
ax=fig.add_subplot(231,projection="3d")
ax.plot_surface(x_mesh_d1,y_mesh_d1_sls,z_mesh_d1, color="blue",label='Plane estimated using standard least square method for pc1.csv')
ax.set_title ("Plane estimated using standard least square for pc1.csv'")
#ax.plot_surface(x_mesh_d1,y_mesh_d1_tls,z_mesh_d1, color="red",label='Plane estimated using total least square method')
ax.scatter3D(x_1,y_1,z_1, c= "green", label='raw data')
#plt.legend(loc="upper right")
ax=fig.add_subplot(232,projection="3d")
ax.plot_surface(x_mesh_d1,y_mesh_d1_tls,z_mesh_d1, color="red",label='Plane estimated using total least square method for pc1.csv')
ax.set_title ("Plane estimated using total least square for pc1.csv")
ax.scatter3D(x_1,y_1,z_1, c= "green", label='raw data')

### 2.2 (b) RANSAC plotting for pc1.csv data
ax=fig.add_subplot(233,projection="3d")
ax.plot_surface(x_mesh_d1,y_mesh_d1_ransac,z_mesh_d1,color="blue",label='plane estimated using ransac')
ax.scatter3D(x_1,y_1,z_1, c='green',label='raw data')
ax.set_title("Plane estimated using RANSAC for pc1.csv")


### Plotting the x,y,z in plane from data set pc2.csv
ax=fig.add_subplot(234,projection="3d")
ax.plot_surface(x_mesh_d2,y_mesh_d2_sls,z_mesh_d2, color="blue",label='Plane estimated using standard least square for pc2.csv')
ax.set_title ("Plane estimated using standard least square for pc2.csv")
#ax.plot_surface(x_mesh_d1,y_mesh_d1_tls,z_mesh_d1, color="red",label='Plane estimated using total least square method')
ax.scatter3D(x_2,y_2,z_2, c= "green", label='raw data')
#plt.legend(loc="upper right")
ax=fig.add_subplot(235,projection="3d")
ax.plot_surface(x_mesh_d2,y_mesh_d2_tls,z_mesh_d2, color="red",label='Plane estimated using total least square for pc2.csv')
ax.set_title ("Plane estimated using total least square for pc2.csv")
ax.scatter3D(x_2,y_2,z_2, c= "green", label='raw data')

### 2.2 (b) RANSAC plotting for pc2.csv data
ax=fig.add_subplot(236,projection="3d")
ax.plot_surface(x_mesh_d2,y_mesh_d2_ransac,z_mesh_d2,color="blue",label='plane estimated using ransac')
ax.scatter3D(x_2,y_2,z_2, c='green',label='raw data')
ax.set_title("Plane estimated using RANSAC for pc2.csv")

plt.show()




