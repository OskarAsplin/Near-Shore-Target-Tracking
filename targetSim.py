import numpy as np
import math
import random
import matplotlib.pyplot as plt
import trajectory_tools
#import tracking


# Global constants
clut_h = 0.0001
clut_l = 0.00001

# Initialized target
x0 = np.array([[-300], [4], [-200], [5]])
cov0 = np.diag([10, 0.5, 10, 0.5])
x0_est = x0 + np.dot(np.linalg.cholesky(cov0),np.random.normal(size=[4,1]))
v_max = 25

# Time for simulation
dt = 2.5
t_end = 300
time = np.arange(1,t_end,dt)

# Area of simulation
x_lim = [-2000, 2000]
y_lim = [-2000, 2000]
V = (x_lim[1] - x_lim[0])*(y_lim[1] - y_lim[0])   # Area (m^2)

# Kalman filter stuff
q = 0.25                     # Process noise strength squared
r = 50
F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
R = r*np.identity(2)
G = np.array([[dt**2/2], [dt]])
Q = q*np.append(np.append(G*G.T, np.zeros([2,2]),axis=1),
    np.append(np.zeros([2,2]), G*G.T,axis=1),axis=0)
K = len(time)
gamma_g = 9.21               # Gate threshold
c = 2                        # Normalization constant

#PDA constants
PG = 0.99
PD = 0.9

# Empty state and covariance vectors
x_true = np.zeros([4,K])
x_est_prior = np.zeros([4,K])
x_est_posterior = np.zeros([4,K])
cov_prior = np.zeros([4,4,K])
cov_posterior = np.zeros([4,4,K])

# Measurement vectors
z_true = np.zeros([2, K])
z_gate = np.empty((2,0))

# -----------------------------------------------

random.seed(a=None)

# Generate target trajectory - random turns, constant velocity
traj_tools = trajectory_tools.TrajectoryChange()
for k, t in enumerate(time):
    if k == 0:
        x_true[:,k] = x0.T
    else:
        x_true[:,k] = F.dot(x_true[:, k - 1])
        x_true[:,k] = traj_tools.randomize_direction(x_true[:,k]).reshape(4)

# Main loop
for k, t in enumerate(time):
    # Dynamic model
    if k == 0:
        x_est_prior[:,k] = x0_est.T
        cov_prior[:,:,k] = cov0
    else:
        x_est_prior[:,k] = F.dot(x_est_posterior[:,k-1])
        cov_prior[:,:,k] = F.dot(cov_posterior[:,:,k-1]).dot(F.T)+Q
    
    # Generate measurement for real target
    noise = np.dot(np.linalg.cholesky(R),np.random.normal(size=[2,1]))
    z_true[:,k] = H.dot(x_true[:,k])+noise.T
    
    # Add clutter measurements and signal strength
    lambda1 = clut_l + (clut_h - clut_l)*random.random()   # Clutter density
    num_clut = math.floor(lambda1*V)              # Number of clutter points
    
    z_all = np.array([np.append(z_true[0,k], np.random.randint(low=x_lim[0], high=x_lim[1],size=num_clut)),
        np.append(z_true[1,k], np.random.randint(low=y_lim[0], high=y_lim[1],size=num_clut)),
        np.random.random(num_clut+1)])
    
    # Test signal strength
    z_strength = np.empty((0,2))

    for i in range(len(z_all[0,:])):
        if (z_all[2,i] < PD):
            z_strength = np.append(z_strength, [z_all[0:2,i]], axis=0)

    # Find measurements within validation region
    S = H.dot(cov_prior[:,:,k]).dot(H.T)+R           # Covariance of the innovation
    W = cov_prior[:,:,k].dot(H.T)             # Gain
    W = W.dot(np.linalg.inv(S))
    m_k = 0
    beta_i = np.array([0])
    v_i = np.array([[0], [0]])
    for i in range(len(z_strength[:,0])):
        v_ik = z_strength[i] - np.dot(H,x_est_prior[:,k])       # Measurement innovation
        NIS_temp = v_ik.T.dot(np.linalg.inv(S)).dot(v_ik)
        if NIS_temp < gamma_g:                           # Within validation region
            z_gate = np.append(z_gate, [[z_strength[i,0]], [z_strength[i,1]]], axis=1)
            v_i = np.append(v_i, [[v_ik[0]], [v_ik[1]]], axis=1)
            beta_i = np.append(beta_i, np.exp(-0.5*NIS_temp))
            m_k += 1

    beta_i[0] = 2*(1-PD*PG)*m_k/(gamma_g)
    if beta_i[0] != 0:
        beta_i = beta_i/np.sum(beta_i)                  # Normalize
    
    if m_k != 0:
        v_k = np.array([[v_i[0,1:].dot(beta_i[1:])],
            [v_i[1,1:].dot(beta_i[1:])]])
    else:
        v_k = np.array([[0], [0]])
    
    # Covariance of correct measurement and SOI
    P_c = cov_prior[:,:,k] - W.dot(S).dot(W.T)
    part_result = 0
    for i in range(1,len(beta_i)):
        temp = v_i[:,i][np.newaxis]
        part_result = part_result + beta_i[i]*(temp.T*temp)
    part_result -= v_k.dot(v_k.T)
    SOI = W.dot(part_result).dot(W.T)

    temp = np.transpose([x_est_prior[:,k]])+W.dot(v_k)
    x_est_posterior[:,k] = temp.reshape(4)
    cov_posterior[:,:,k] = beta_i[0]*cov_prior[:,:,k]+(1-beta_i[0])*P_c+SOI

# -----------------------------------------------

plt.plot(x_true[2,:], x_true[0,:],'r', label='True trajectory')
plt.plot(z_true[1,:], z_true[0,:], 'o', fillstyle='none', markeredgecolor='xkcd:wheat', label='Data from target')
plt.plot(z_gate[1,:], z_gate[0,:], 'kx', label='Data within gate')
plt.plot(x_est_posterior[2,:], x_est_posterior[0,:], 'b--', label='Posterior estimation', linewidth=2)

plt.xlabel('East[m]')
plt.ylabel('North[m]')
plt.title('Track position with sample rate: 1/s')
plt.legend()


plt.show()
