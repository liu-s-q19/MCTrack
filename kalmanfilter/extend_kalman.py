# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

"""
    Extended Kalman Filter for MOT
"""

import numpy as np

from kalmanfilter.base_kalman import KF_Base
from utils.utils import norm_radian, norm_realative_radian
from filterpy.kalman import unscented_transform as UT
from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints
from autograd import hessian, jacobian
from scipy.optimize import minimize
from .utils import (cal_mean, cal_mean_mc, is_positive_semidefinite,
                    kl_divergence)

class KF_YAW(KF_Base):
    """
    kalman filter for yaw in tracking objects
    """

    def __init__(self, n=2, m=2, dt=0.1, P=None, Q=None, R=None, init_x=None, cfg=None):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=cfg)
        self.dt = dt
        self.JH = np.matrix([[1.0, 0.0], [1.0, 0.0]])
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)

    def f(self, x, dt=None):
        # State-transition function is identity
        dt = self.dt if dt is None else dt
        x_fil = np.array(x)
        x_fil[0] = x[0] + dt * x[1]
        x_fil[0] = norm_radian(x_fil[0])
        return x_fil

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        dt = self.dt
        F = np.matrix([[1.0, dt], [0.0, 1.0]])
        return F

    def h(self, x):
        # Observation function is identity
        return self.JH * x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH

    def step(self, z, dt=None):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------
        dt = self.dt if dt is None else dt
        # shape of x :[n,1]  only norm angle,not angle_ratio
        self.x[0][0] = norm_radian(self.x[0][0])
        self.x = self.f(self.x, dt)
        self.x[0][0] = norm_radian(self.x[0][0])

        self.F = self.getF(self.x)
        self.P = self.F * self.P * self.F.T + self.Q

        # Update -----------------------------------------------------
        G = self.P * self.H.T * np.linalg.inv(self.H * self.P * self.H.T + self.R)
        hx = self.h(self.x)  # shape:(m,1)

        hx = norm_radian(hx).reshape(-1, 1)
        z = np.array(z).reshape(-1, 1)  # (m,1)
        z = norm_radian(np.array(z)).reshape(-1, 1)
        info_gain = z - hx
        info_gain = norm_realative_radian(info_gain).reshape(-1, 1)

        self.x += G * info_gain
        self.P = (self.I - np.matmul(G, self.H)) * self.P
        self.x[0][0] = norm_radian(self.x[0][0])
        return np.array(self.x.reshape(self.n))


class KF_SIZE(KF_Base):
    """
    An EKF for tracking
    """

    def __init__(
        self, n=4, m=2, dt=None, P=None, Q=None, R=None, init_x=None, cfg=None
    ):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=None)

        self.dt = dt
        self.JH = np.matrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)

    def f(self, x):
        # State-transition function is identity

        # dt = dt if dt else self.dt
        x_fil = np.array(x)
        x_fil[0] = x[0] + self.dt * x[2]
        x_fil[1] = x[1] + self.dt * x[3]
        return x_fil

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        a13 = self.dt
        a14 = 0
        a23 = 0
        a24 = self.dt
        F = np.matrix(
            [
                [1.0, 0.0, a13, a14],
                [0.0, 1.0, a23, a24],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F

    def h(self, x):
        # Observation function is identity
        return self.JH * x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH

    def update(self, z):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------
        self.x = self.f(self.x)
        self.F = self.getF(self.x)
        self.P = self.F * self.P * self.F.T + self.Q

        # Update -----------------------------------------------------
        G = self.P * self.H.T * np.linalg.inv(self.H * self.P * self.H.T + self.R)
        self.x += G * (np.array(z) - self.h(self.x).T).T
        #         self.P = np.matmul(self.I - np.matmul(G, self.H), self.P)
        self.P = (self.I - np.matmul(G, self.H)) * self.P
        # return self.x.asarray()
        return np.array(self.x.reshape(self.n))


class EKF_CV(KF_Base):
    """
    An EKF for tracking
    """

    def __init__(
        self, n=4, m=2, dt=None, P=None, Q=None, R=None, init_x=None, cfg=None
    ):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=None)

        self.dt = dt
        self.m = m
        self.JH = np.matrix(np.eye(self.m, 4))
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)

    def f(self, x):
        # State-transition function is identity

        # dt = dt if dt else self.dt
        x_fil = np.array(x)
        x_fil[0] = x[0] + self.dt * x[2]
        x_fil[1] = x[1] + self.dt * x[3]
        return x_fil

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        a13 = self.dt
        a14 = 0
        a23 = 0
        a24 = self.dt
        F = np.matrix(
            [
                [1.0, 0.0, a13, a14],
                [0.0, 1.0, a23, a24],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F

    def h(self, x):
        # Observation function is identity
        return self.JH * x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH

    def update(self, z):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------
        self.x = self.f(self.x)
        self.F = self.getF(self.x)
        self.P = self.F * self.P * self.F.T + self.Q

        # Update -----------------------------------------------------
        G = self.P * self.H.T * np.linalg.inv(self.H * self.P * self.H.T + self.R)
        self.x += G * (np.array(z) - self.h(self.x).T).T
        #         self.P = np.matmul(self.I - np.matmul(G, self.H), self.P)
        self.P = (self.I - np.matmul(G, self.H)) * self.P
        # return self.x.asarray()
        return np.array(self.x.reshape(self.n))

class WeightEKF_CV(KF_Base):
    """
    An EKF for tracking
    """

    def __init__(
        self, n=4, m=2, dt=None, P=None, Q=None, R=None, init_x=None, cfg=None
    ):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=None)

        self.dt = dt
        self.m = m
        self.JH = np.matrix(np.eye(self.m, 4))
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)

    def f(self, x):
        # State-transition function is identity

        # dt = dt if dt else self.dt
        x_fil = np.array(x)
        x_fil[0] = x[0] + self.dt * x[2]
        x_fil[1] = x[1] + self.dt * x[3]
        return x_fil

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        a13 = self.dt
        a14 = 0
        a23 = 0
        a24 = self.dt
        F = np.matrix(
            [
                [1.0, 0.0, a13, a14],
                [0.0, 1.0, a23, a24],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F

    def h(self, x):
        # Observation function is identity
        return self.JH * x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH

    def update(self, z):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------
        self.x = self.f(self.x)
        self.F = self.getF(self.x)
        self.P = self.F * self.P * self.F.T + self.Q

        # Update -----------------------------------------------------
        soft_threshold = 10.0
        err = np.array(z) - self.h(self.x).T
        weight_term = soft_threshold ** 2 / (soft_threshold ** 2 + np.inner(err, err))
        Rt = self.R * weight_term
        G = self.P * self.H.T * np.linalg.inv(self.H * self.P * self.H.T + Rt)
        self.x += G * (np.array(z) - self.h(self.x).T).T
        #         self.P = np.matmul(self.I - np.matmul(G, self.H), self.P)
        self.P = (self.I - np.matmul(G, self.H)) * self.P
        # return self.x.asarray()
        return np.array(self.x.reshape(self.n))

class NANO_CV(KF_Base):
    """
    An EKF for tracking
    """

    def __init__(
        self, n=4, m=2, dt=None, P=None, Q=None, R=None, init_x=None, cfg=None, 
        loss_type='log_likelihood_loss', init_type='iekf', derivate_type='partical', 
        iekf_max_iter=1, n_iterations=1, delta=5, c=5, beta=1e-4, lr=1.0, threshold=1e-4
    ):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=None)

        self.dt = dt
        self.m = m
        self.dim_x = n
        self.dim_y = m
        self.JH = np.eye(self.m)
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)
        self.lr = lr
        self.threshold = threshold
        self.n_iterations = n_iterations
        self.points = MerweScaledSigmaPoints(self.dim_x, alpha=0.1, beta=2.0, kappa=1.0)
        self.x = self.x.reshape(self.dim_x,)
        self.x_prior = self.x
        self.P_prior = self.P
        self.x_post = self.x
        self.P_post = self.P
        self.init_type = init_type
        self.derivate_type = derivate_type
        self.iekf_max_iter = iekf_max_iter
        self.delta = delta
        self.c = c
        self.beta = beta
        self._I = np.eye(self.dim_x)
        # print('loss_func', loss_type)
        if loss_type == 'huber_loss':
            self.loss_func = self.pseudo_huber_loss
        elif loss_type == 'weighted_loss':
            self.loss_func = self.weighted_log_likelihood_loss
        elif loss_type == 'beta_loss':
            self.loss_func = self.beta_likelihood_loss
        elif loss_type == 'loglikelihood_loss':
            self.loss_func = self.log_likelihood_loss
        else:
            raise ValueError('loss_type no found')

    def log_likelihood_loss(self, x, y):
        return 0.5 * np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))
    
    def pseudo_huber_loss(self, x, y):
        delta = self.delta
        mse_loss = np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))
        return delta**2 * (np.sqrt(1 + mse_loss / delta**2) - 1)
    
    def weighted_log_likelihood_loss(self, x, y):
        c = self.c
        mse_loss = 0.5 * np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))
        weight = 1/(1 + mse_loss / c**2)
        return weight * mse_loss
    
    def beta_likelihood_loss(self, x, y):
        beta = self.beta
        R_inv = np.linalg.inv(self.R)
        det_R = np.linalg.det(self.R)
        return 1 / ((beta + 1)**1.5*(2*np.pi)**(self.dim_y*beta/2)) * det_R**(beta / 2) \
                - (1 / beta + 1) / ((2 * np.pi)**(beta*self.dim_y/2) * det_R**(beta/2)) * np.exp(-0.5*beta*(y-self.h(x)).T @ R_inv @ (y-self.h(x)))

    def loss_func_jacobian(self, x, y):
        # cal jacobian of loss function
        return jacobian(lambda x: self.loss_func(x, y))(x)
        
    def loss_func_hessian(self, x, y):
        # cal hessian of loss function
        return hessian(lambda x: self.loss_func(x, y))(x)
    
    def loss_func_hessian_diff(self, x, y, epsilon=5e-5):
        n = len(x)
        Hessian = np.zeros((n, n))
        f = self.loss_func
        fx = f(x, y)
        
        for i in range(n):
            for j in range(i, n):
                x_ij = x.copy()
                x_ij[i] += epsilon
                x_ij[j] += epsilon
                fij = f(x_ij, y)
                
                x_i = x.copy()
                x_i[i] += epsilon
                fi = f(x_i, y)
                
                x_j = x.copy()
                x_j[j] += epsilon
                fj = f(x_j, y)
                
                Hessian[i, j] = (fij - fi - fj + fx) / (epsilon**2)
                Hessian[j, i] = Hessian[i, j]
                
        return Hessian
    
    def map_loss(self, x_prior, P_prior, x_posterior, y):
        l1 = 0.5 * (x_posterior - x_prior).T @ np.linalg.inv(P_prior) @ (x_posterior - x_prior) 
        l2 = self.loss_func(x_posterior, y)
        return l1 + l2
    
    def f(self, x):
        # State-transition function is identity

        # dt = dt if dt else self.dt
        x_fil = np.array(x)
        x_fil[0] = x[0] + self.dt * x[2]
        x_fil[1] = x[1] + self.dt * x[3]
        return x_fil

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        a13 = self.dt
        a14 = 0
        a23 = 0
        a24 = self.dt
        F = np.matrix(
            [
                [1.0, 0.0, a13, a14],
                [0.0, 1.0, a23, a24],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F

    def h(self, x):
        # Observation function is identity
        return self.H @ x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH

    def update_init(self, y, x_prior, P_prior):
        # Laplace Approximation
        loss = lambda x_posterior: self.map_loss(x_prior, P_prior, x_posterior, y)
        x_hat_posterior = minimize(loss, x0=x_prior, method='BFGS').x
        P_posterior_inv = hessian(lambda x: self.map_loss(x_prior, P_prior, x, y))(x_hat_posterior)
        return x_hat_posterior, P_posterior_inv

    def update_iekf_init(self, y, x_prior, P_prior, max_iter=1):
        # Iterated Extended Kalman Filter (IEKF) for Maximum A Posteriori (MAP)
        x_hat = x_prior
        for i in range(max_iter):
            H = self.H
            hx = self.h(x_hat)
            v = y - hx - H @ (x_prior - x_hat)
            PHT = P_prior @ H.T
            S = H @ PHT + self.R
            K = PHT @ np.linalg.inv(S)
            x_hat = x_prior + K @ v
        
        x_hat_posterior = x_hat
        I_KH = self._I - K @ H
        P_posterior = (I_KH @ P_prior @ I_KH.T) + (K @ self.R @ K.T)
        P_posterior_inv = np.linalg.inv(P_posterior)
        
        return x_hat_posterior, P_posterior_inv

    def predict(self):
        
        sigmas = self.points.sigma_points(self.x, self.P)

        self.sigmas_f = np.zeros((len(sigmas), self.dim_x))
        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = self.f(s)        
        self.x, self.P = UT(self.sigmas_f, self.points.Wm, self.points.Wc, self.Q)
        
        is_positive_semidefinite(self.P)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        return self.x
    
    def update(self, z):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------

        # Update -----------------------------------------------------
        lr = self.lr
        n_iterations = self.n_iterations    
        # Initialize the first iteration step
        x_hat_prior = self.x.copy()
        P_inv_prior = np.linalg.inv(self.P).copy()
        if self.init_type == 'prior':
            x_hat, P_inv = x_hat_prior, P_inv_prior
        elif self.init_type == 'laplace':
            x_hat, P_inv = self.update_init(z, x_hat_prior, self.P.copy())
        elif self.init_type == 'iekf':
            x_hat, P_inv = self.update_iekf_init(z, x_hat_prior, self.P.copy(), max_iter=self.iekf_max_iter)
        else:
            raise ValueError('init_type should be prior or laplace')
        
        is_positive_semidefinite(P_inv)
        
        for _ in range(n_iterations):         
            P = np.linalg.inv(P_inv)
            is_positive_semidefinite(P)

            if self.derivate_type == 'stein':
                E_hessian = P_inv @ cal_mean(lambda x: np.outer(x-x_hat, x-x_hat)*self.loss_func(x, z), x_hat, P, self.points) @ P_inv \
                            - cal_mean(lambda x: self.loss_func(x, z), x_hat, P, self.points) * P_inv
                P_inv_next = P_inv_prior - lr * E_hessian
                is_positive_semidefinite(P_inv_next)
                P_next = np.linalg.inv(P_inv_next)
                x_hat_next = x_hat - lr*(P_next @ P_inv @ cal_mean(lambda x: (x - x_hat) * self.loss_func(x, z), x_hat, P, self.points) - P_next @ P_inv_prior @ (x_hat - x_hat_prior))

            elif self.derivate_type == 'direct':
                P_inv_next = P_inv_prior + lr*cal_mean(lambda x: self.loss_func_hessian_diff(x, z), x_hat, P, self.points)
                is_positive_semidefinite(P_inv_next)
                P_next = np.linalg.inv(P_inv_next)
                x_hat_next = x_hat - lr*(P_next @ cal_mean(lambda x: self.loss_func_jacobian(x, z), x_hat, P, self.points) - P_next @ P_inv_prior @ (x_hat - x_hat_prior))

            elif self.derivate_type == 'partical':
                _, P_inv_next = self.update_iekf_init(z, x_hat, P, max_iter=self.iekf_max_iter)
                P_next = np.linalg.inv(P_inv_next)
                x_hat_next = x_hat - lr*(P_next @ P_inv @ cal_mean(lambda x: (x - x_hat) * self.loss_func(x, z), x_hat, P, self.points) - P_next @ P_inv_prior @ (x_hat - x_hat_prior))
            else:
                raise ValueError('derivate_type no found')
                
            kld = kl_divergence(x_hat, P, x_hat_next, P_next)
            if kld < self.threshold:
                P_inv = P_inv_next.copy()
                x_hat = x_hat_next.copy()
                break

            P_inv = P_inv_next.copy()
            x_hat = x_hat_next.copy()
            
        self.x = x_hat
        self.P = np.linalg.inv(P_inv)

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        
        return np.array(self.x.reshape(self.n))
     
class NANO_YAW(KF_Base):
    """
    An EKF for tracking
    """

    def __init__(
        self, n=2, m=2, dt=0.1, P=None, Q=None, R=None, init_x=None, cfg=None, 
        loss_type='log_likelihood_loss', init_type='iekf', derivate_type='partical', 
        iekf_max_iter=1, n_iterations=1, delta=5, c=5, beta=1e-4, lr=1.0, threshold=1e-4
    ):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=None)

        self.dt = dt
        self.m = m
        self.dim_x = n
        self.dim_y = m
        self.JH = np.array([[1.0, 0.0], [1.0, 0.0]])
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)
        self.lr = lr
        self.threshold = threshold
        self.n_iterations = n_iterations
        self.points = MerweScaledSigmaPoints(self.dim_x, alpha=0.1, beta=2.0, kappa=1.0)
        self.x = self.x.reshape(self.dim_x,)
        self.x_prior = self.x
        self.P_prior = self.P
        self.x_post = self.x
        self.P_post = self.P
        self.init_type = init_type
        self.derivate_type = derivate_type
        self.iekf_max_iter = iekf_max_iter
        self.delta = delta
        self.c = c
        self.beta = beta
        self._I = np.eye(self.dim_x)
        # print('loss_func', loss_type)
        if loss_type == 'huber_loss':
            self.loss_func = self.pseudo_huber_loss
        elif loss_type == 'weighted_loss':
            self.loss_func = self.weighted_log_likelihood_loss
        elif loss_type == 'beta_loss':
            self.loss_func = self.beta_likelihood_loss
        elif loss_type == 'loglikelihood_loss':
            self.loss_func = self.log_likelihood_loss
        else:
            raise ValueError('loss_type no found')

    def log_likelihood_loss(self, x, y):
        return 0.5 * np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))
    
    def pseudo_huber_loss(self, x, y):
        delta = self.delta
        mse_loss = np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))
        return delta**2 * (np.sqrt(1 + mse_loss / delta**2) - 1)
    
    def weighted_log_likelihood_loss(self, x, y):
        c = self.c
        mse_loss = 0.5 * np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))
        weight = 1/(1 + mse_loss / c**2)
        return weight * mse_loss
    
    def beta_likelihood_loss(self, x, y):
        beta = self.beta
        R_inv = np.linalg.inv(self.R)
        det_R = np.linalg.det(self.R)
        return 1 / ((beta + 1)**1.5*(2*np.pi)**(self.dim_y*beta/2)) * det_R**(beta / 2) \
                - (1 / beta + 1) / ((2 * np.pi)**(beta*self.dim_y/2) * det_R**(beta/2)) * np.exp(-0.5*beta*(y-self.h(x)).T @ R_inv @ (y-self.h(x)))

    def loss_func_jacobian(self, x, y):
        # cal jacobian of loss function
        return jacobian(lambda x: self.loss_func(x, y))(x)
        
    def loss_func_hessian(self, x, y):
        # cal hessian of loss function
        return hessian(lambda x: self.loss_func(x, y))(x)
    
    def loss_func_hessian_diff(self, x, y, epsilon=5e-5):
        n = len(x)
        Hessian = np.zeros((n, n))
        f = self.loss_func
        fx = f(x, y)
        
        for i in range(n):
            for j in range(i, n):
                x_ij = x.copy()
                x_ij[i] += epsilon
                x_ij[j] += epsilon
                fij = f(x_ij, y)
                
                x_i = x.copy()
                x_i[i] += epsilon
                fi = f(x_i, y)
                
                x_j = x.copy()
                x_j[j] += epsilon
                fj = f(x_j, y)
                
                Hessian[i, j] = (fij - fi - fj + fx) / (epsilon**2)
                Hessian[j, i] = Hessian[i, j]
                
        return Hessian
    
    def map_loss(self, x_prior, P_prior, x_posterior, y):
        l1 = 0.5 * (x_posterior - x_prior).T @ np.linalg.inv(P_prior) @ (x_posterior - x_prior) 
        l2 = self.loss_func(x_posterior, y)
        return l1 + l2
    
    def f(self, x, dt=None):
        dt = self.dt if dt is None else dt
        x_fil = np.array(x)
        x_fil[0] = x[0] + dt * x[1]
        x_fil[0] = norm_radian(x_fil[0])
        return x_fil

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        dt = self.dt
        F = np.matrix([[1.0, dt], [0.0, 1.0]])
        return F

    def h(self, x):
        # Observation function is identity
        return self.H @ x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH

    def update_init(self, y, x_prior, P_prior):
        # Laplace Approximation
        loss = lambda x_posterior: self.map_loss(x_prior, P_prior, x_posterior, y)
        x_hat_posterior = minimize(loss, x0=x_prior, method='BFGS').x
        P_posterior_inv = hessian(lambda x: self.map_loss(x_prior, P_prior, x, y))(x_hat_posterior)
        return x_hat_posterior, P_posterior_inv

    def update_iekf_init(self, y, x_prior, P_prior, max_iter=1):
        # Iterated Extended Kalman Filter (IEKF) for Maximum A Posteriori (MAP)
        x_hat = x_prior
        for i in range(max_iter):
            H = self.H
            hx = self.h(x_hat)
            v = y - hx - H @ (x_prior - x_hat)
            PHT = P_prior @ H.T
            S = H @ PHT + self.R
            K = PHT @ np.linalg.inv(S)
            x_hat = x_prior + K @ v
        
        x_hat_posterior = x_hat
        I_KH = self._I - K @ H
        P_posterior = (I_KH @ P_prior @ I_KH.T) + (K @ self.R @ K.T)
        P_posterior_inv = np.linalg.inv(P_posterior)
        
        return x_hat_posterior, P_posterior_inv

    def predict(self):
        
        sigmas = self.points.sigma_points(self.x, self.P)

        self.sigmas_f = np.zeros((len(sigmas), self.dim_x))
        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = self.f(s)        
        
        self.x, self.P = UT(self.sigmas_f, self.points.Wm, self.points.Wc, self.Q)
        
        is_positive_semidefinite(self.P)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        return self.x
    
    def update(self, z):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------

        # Update -----------------------------------------------------
        lr = self.lr
        n_iterations = self.n_iterations    
        # Initialize the first iteration step
        x_hat_prior = self.x.copy()
        P_inv_prior = np.linalg.inv(self.P).copy()
        if self.init_type == 'prior':
            x_hat, P_inv = x_hat_prior, P_inv_prior
        elif self.init_type == 'laplace':
            x_hat, P_inv = self.update_init(z, x_hat_prior, self.P.copy())
        elif self.init_type == 'iekf':
            x_hat, P_inv = self.update_iekf_init(z, x_hat_prior, self.P.copy(), max_iter=self.iekf_max_iter)
        else:
            raise ValueError('init_type should be prior or laplace')
        
        is_positive_semidefinite(P_inv)
        
        for _ in range(n_iterations):         
            P = np.linalg.inv(P_inv)
            is_positive_semidefinite(P)

            if self.derivate_type == 'stein':
                E_hessian = P_inv @ cal_mean(lambda x: np.outer(x-x_hat, x-x_hat)*self.loss_func(x, z), x_hat, P, self.points) @ P_inv \
                            - cal_mean(lambda x: self.loss_func(x, z), x_hat, P, self.points) * P_inv
                P_inv_next = P_inv_prior - lr * E_hessian
                is_positive_semidefinite(P_inv_next)
                P_next = np.linalg.inv(P_inv_next)
                x_hat_next = x_hat - lr*(P_next @ P_inv @ cal_mean(lambda x: (x - x_hat) * self.loss_func(x, z), x_hat, P, self.points) - P_next @ P_inv_prior @ (x_hat - x_hat_prior))

            elif self.derivate_type == 'direct':
                P_inv_next = P_inv_prior + lr*cal_mean(lambda x: self.loss_func_hessian_diff(x, z), x_hat, P, self.points)
                is_positive_semidefinite(P_inv_next)
                P_next = np.linalg.inv(P_inv_next)
                x_hat_next = x_hat - lr*(P_next @ cal_mean(lambda x: self.loss_func_jacobian(x, z), x_hat, P, self.points) - P_next @ P_inv_prior @ (x_hat - x_hat_prior))

            elif self.derivate_type == 'partical':
                _, P_inv_next = self.update_iekf_init(z, x_hat, P, max_iter=self.iekf_max_iter)
                P_next = np.linalg.inv(P_inv_next)
                x_hat_next = x_hat - lr*(P_next @ P_inv @ cal_mean(lambda x: (x - x_hat) * self.loss_func(x, z), x_hat, P, self.points) - P_next @ P_inv_prior @ (x_hat - x_hat_prior))
            else:
                raise ValueError('derivate_type no found')

            kld = kl_divergence(x_hat, P, x_hat_next, P_next)
            if kld < self.threshold:
                P_inv = P_inv_next.copy()
                x_hat = x_hat_next.copy()
                break

            P_inv = P_inv_next.copy()
            x_hat = x_hat_next.copy()
            
        self.x = x_hat
        self.P = np.linalg.inv(P_inv)

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        
        return np.array(self.x.reshape(self.n))
     
class NANO_SIZE(KF_Base):
    """
    An EKF for tracking
    """

    def __init__(
        self, n=4, m=2, dt=None, P=None, Q=None, R=None, init_x=None, cfg=None, 
        loss_type='log_likelihood_loss', init_type='iekf', derivate_type='partical', 
        iekf_max_iter=1, n_iterations=1, delta=5, c=5, beta=1e-4, lr=1.0, threshold=1e-4
    ):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=None)

        self.dt = dt
        self.m = m
        self.dim_x = n
        self.dim_y = m
        self.JH = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)
        self.lr = lr
        self.threshold = threshold
        self.n_iterations = n_iterations
        self.points = MerweScaledSigmaPoints(self.dim_x, alpha=0.1, beta=2.0, kappa=1.0)
        self.x = self.x.reshape(self.dim_x,)
        self.x_prior = self.x
        self.P_prior = self.P
        self.x_post = self.x
        self.P_post = self.P
        self.init_type = init_type
        self.derivate_type = derivate_type
        self.iekf_max_iter = iekf_max_iter
        self.delta = delta
        self.c = c
        self.beta = beta
        self._I = np.eye(self.dim_x)
        # print('loss_func', loss_type)
        if loss_type == 'huber_loss':
            self.loss_func = self.pseudo_huber_loss
        elif loss_type == 'weighted_loss':
            self.loss_func = self.weighted_log_likelihood_loss
        elif loss_type == 'beta_loss':
            self.loss_func = self.beta_likelihood_loss
        elif loss_type == 'loglikelihood_loss':
            self.loss_func = self.log_likelihood_loss
        else:
            raise ValueError('loss_type no found')

    def log_likelihood_loss(self, x, y):
        return 0.5 * np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))
    
    def pseudo_huber_loss(self, x, y):
        delta = self.delta
        mse_loss = np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))
        return delta**2 * (np.sqrt(1 + mse_loss / delta**2) - 1)
    
    def weighted_log_likelihood_loss(self, x, y):
        c = self.c
        mse_loss = 0.5 * np.dot(y - self.h(x), np.dot(np.linalg.inv(self.R), y - self.h(x)))
        weight = 1/(1 + mse_loss / c**2)
        return weight * mse_loss
    
    def beta_likelihood_loss(self, x, y):
        beta = self.beta
        R_inv = np.linalg.inv(self.R)
        det_R = np.linalg.det(self.R)
        return 1 / ((beta + 1)**1.5*(2*np.pi)**(self.dim_y*beta/2)) * det_R**(beta / 2) \
                - (1 / beta + 1) / ((2 * np.pi)**(beta*self.dim_y/2) * det_R**(beta/2)) * np.exp(-0.5*beta*(y-self.h(x)).T @ R_inv @ (y-self.h(x)))

    def loss_func_jacobian(self, x, y):
        # cal jacobian of loss function
        return jacobian(lambda x: self.loss_func(x, y))(x)
        
    def loss_func_hessian(self, x, y):
        # cal hessian of loss function
        return hessian(lambda x: self.loss_func(x, y))(x)
    
    def loss_func_hessian_diff(self, x, y, epsilon=5e-5):
        n = len(x)
        Hessian = np.zeros((n, n))
        f = self.loss_func
        fx = f(x, y)
        
        for i in range(n):
            for j in range(i, n):
                x_ij = x.copy()
                x_ij[i] += epsilon
                x_ij[j] += epsilon
                fij = f(x_ij, y)
                
                x_i = x.copy()
                x_i[i] += epsilon
                fi = f(x_i, y)
                
                x_j = x.copy()
                x_j[j] += epsilon
                fj = f(x_j, y)
                
                Hessian[i, j] = (fij - fi - fj + fx) / (epsilon**2)
                Hessian[j, i] = Hessian[i, j]
                
        return Hessian
    
    def map_loss(self, x_prior, P_prior, x_posterior, y):
        l1 = 0.5 * (x_posterior - x_prior).T @ np.linalg.inv(P_prior) @ (x_posterior - x_prior) 
        l2 = self.loss_func(x_posterior, y)
        return l1 + l2
    
    def f(self, x):
        # State-transition function is identity

        # dt = dt if dt else self.dt
        x_fil = np.array(x)
        x_fil[0] = x[0] + self.dt * x[2]
        x_fil[1] = x[1] + self.dt * x[3]
        return x_fil

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        a13 = self.dt
        a14 = 0
        a23 = 0
        a24 = self.dt
        F = np.matrix(
            [
                [1.0, 0.0, a13, a14],
                [0.0, 1.0, a23, a24],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F

    def h(self, x):
        # Observation function is identity
        return self.H @ x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH

    def update_init(self, y, x_prior, P_prior):
        # Laplace Approximation
        loss = lambda x_posterior: self.map_loss(x_prior, P_prior, x_posterior, y)
        x_hat_posterior = minimize(loss, x0=x_prior, method='BFGS').x
        P_posterior_inv = hessian(lambda x: self.map_loss(x_prior, P_prior, x, y))(x_hat_posterior)
        return x_hat_posterior, P_posterior_inv

    def update_iekf_init(self, y, x_prior, P_prior, max_iter=1):
        # Iterated Extended Kalman Filter (IEKF) for Maximum A Posteriori (MAP)
        x_hat = x_prior
        for i in range(max_iter):
            H = self.H
            hx = self.h(x_hat)
            v = y - hx - H @ (x_prior - x_hat)
            PHT = P_prior @ H.T
            S = H @ PHT + self.R
            K = PHT @ np.linalg.inv(S)
            x_hat = x_prior + K @ v
        
        x_hat_posterior = x_hat
        I_KH = self._I - K @ H
        P_posterior = (I_KH @ P_prior @ I_KH.T) + (K @ self.R @ K.T)
        P_posterior_inv = np.linalg.inv(P_posterior)
        
        return x_hat_posterior, P_posterior_inv

    def predict(self):
        
        sigmas = self.points.sigma_points(self.x, self.P)

        self.sigmas_f = np.zeros((len(sigmas), self.dim_x))
        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = self.f(s)        
        
        self.x, self.P = UT(self.sigmas_f, self.points.Wm, self.points.Wc, self.Q)
        
        is_positive_semidefinite(self.P)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        return self.x
    
    def update(self, z):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------

        # Update -----------------------------------------------------
        lr = self.lr
        n_iterations = self.n_iterations    
        # Initialize the first iteration step
        x_hat_prior = self.x.copy()
        P_inv_prior = np.linalg.inv(self.P).copy()
        if self.init_type == 'prior':
            x_hat, P_inv = x_hat_prior, P_inv_prior
        elif self.init_type == 'laplace':
            x_hat, P_inv = self.update_init(z, x_hat_prior, self.P.copy())
        elif self.init_type == 'iekf':
            x_hat, P_inv = self.update_iekf_init(z, x_hat_prior, self.P.copy(), max_iter=self.iekf_max_iter)
        else:
            raise ValueError('init_type should be prior or laplace')
        
        is_positive_semidefinite(P_inv)
        
        for _ in range(n_iterations):         
            P = np.linalg.inv(P_inv)
            is_positive_semidefinite(P)

            if self.derivate_type == 'stein':
                E_hessian = P_inv @ cal_mean(lambda x: np.outer(x-x_hat, x-x_hat)*self.loss_func(x, z), x_hat, P, self.points) @ P_inv \
                            - cal_mean(lambda x: self.loss_func(x, z), x_hat, P, self.points) * P_inv
                P_inv_next = P_inv_prior - lr * E_hessian
                is_positive_semidefinite(P_inv_next)
                P_next = np.linalg.inv(P_inv_next)
                x_hat_next = x_hat - lr*(P_next @ P_inv @ cal_mean(lambda x: (x - x_hat) * self.loss_func(x, z), x_hat, P, self.points) - P_next @ P_inv_prior @ (x_hat - x_hat_prior))

            elif self.derivate_type == 'direct':
                P_inv_next = P_inv_prior + lr*cal_mean(lambda x: self.loss_func_hessian_diff(x, z), x_hat, P, self.points)
                is_positive_semidefinite(P_inv_next)
                P_next = np.linalg.inv(P_inv_next)
                x_hat_next = x_hat - lr*(P_next @ cal_mean(lambda x: self.loss_func_jacobian(x, z), x_hat, P, self.points) - P_next @ P_inv_prior @ (x_hat - x_hat_prior))

            elif self.derivate_type == 'partical':
                _, P_inv_next = self.update_iekf_init(z, x_hat, P, max_iter=self.iekf_max_iter)
                P_next = np.linalg.inv(P_inv_next)
                x_hat_next = x_hat - lr*(P_next @ P_inv @ cal_mean(lambda x: (x - x_hat) * self.loss_func(x, z), x_hat, P, self.points) - P_next @ P_inv_prior @ (x_hat - x_hat_prior))
            else:
                raise ValueError('derivate_type no found')

            kld = kl_divergence(x_hat, P, x_hat_next, P_next)
            if kld < self.threshold:
                P_inv = P_inv_next.copy()
                x_hat = x_hat_next.copy()
                break

            P_inv = P_inv_next.copy()
            x_hat = x_hat_next.copy()
            
        self.x = x_hat
        self.P = np.linalg.inv(P_inv)

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        
        return np.array(self.x.reshape(self.n))   

class NANO_CTRA(NANO_CV):
    """
    An EKF for tracking
    """

    def __init__(
        self, n=6, m=2, dt=None, P=None, Q=None, R=None, init_x=None, cfg=None, 
        loss_type='log_likelihood_loss', init_type='iekf', derivate_type='partical', 
        iekf_max_iter=1, n_iterations=1, delta=5, c=5, beta=1e-4, lr=1.0, threshold=1e-4
    ):
        NANO_CV.__init__(self, n=n, m=m, dt=dt, P=P, Q=Q, R=R, init_x=init_x, cfg=cfg, 
                         loss_type=loss_type, init_type=init_type, derivate_type=derivate_type, 
                         iekf_max_iter=iekf_max_iter, n_iterations=n_iterations, delta=delta, c=c, beta=beta, lr=lr, threshold=threshold)
        self.dt = dt
        self.JH = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)
        
    def f(self, x):
        # State-transition function is identity
        # [x,y,yaw,v,w,a]

        x_fil = np.array(x).reshape(self.dim_x)
        x_fil[0] = x[0] + (1 / x[4] ** 2) * (
            (x[3] * x[4] + x[5] * x[4] * self.dt) * np.sin(x[2] + x[4] * self.dt)
            + x[5] * np.cos(x[2] + x[4] * self.dt)
            - x[3] * x[4] * np.sin(x[2])
            - x[5] * np.cos(x[2])
        )
        x_fil[1] = x[1] + (1 / x[4] ** 2) * (
            (-x[3] * x[4] - x[5] * x[4] * self.dt) * np.cos(x[2] + x[4] * self.dt)
            + x[5] * np.sin(x[2] + x[4] * self.dt)
            + x[3] * x[4] * np.cos(x[2])
            - x[5] * np.sin(x[2])
        )
        x_fil[2] = x[2]
        return x_fil
        
    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        x = np.array(x).reshape(self.dim_x, 1)
        dt = self.dt
        a13 = (
            (
                -x[4] * x[3] * np.cos(x[2])
                + x[5] * np.sin(x[2])
                - x[5] * np.sin(dt * x[4] + x[2])
                + (dt * x[4] * x[5] + x[4] * x[3]) * np.cos(dt * x[4] + x[2])
            )
            / x[4] ** 2
        ).item(0)

        a14 = (
            (-x[4] * np.sin(x[2]) + x[4] * np.sin(dt * x[4] + x[2])) / x[4] ** 2
        ).item(0)

        a15 = (
            (
                -dt * x[5] * np.sin(dt * x[4] + x[2])
                + dt * (dt * x[4] * x[5] + x[4] * x[3]) * np.cos(dt * x[4] + x[2])
                - x[3] * np.sin(x[2])
                + (dt * x[5] + x[3]) * np.sin(dt * x[4] + x[2])
            )
            / x[4] ** 2
            - 2
            * (
                -x[4] * x[3] * np.sin(x[2])
                - x[5] * np.cos(x[2])
                + x[5] * np.cos(dt * x[4] + x[2])
                + (dt * x[4] * x[5] + x[4] * x[3]) * np.sin(dt * x[4] + x[2])
            )
            / x[4] ** 3
        ).item(0)

        a16 = (
            (
                dt * x[4] * np.sin(dt * x[4] + x[2])
                - np.cos(x[2])
                + np.cos(dt * x[4] + x[2])
            )
            / x[4] ** 2
        ).item(0)

        a23 = (
            (
                -x[4] * x[3] * np.sin(x[2])
                - x[5] * np.cos(x[2])
                + x[5] * np.cos(dt * x[4] + x[2])
                - (-dt * x[4] * x[5] - x[4] * x[3]) * np.sin(dt * x[4] + x[2])
            )
            / x[4] ** 2
        ).item(0)
        a24 = (
            (x[4] * np.cos(x[2]) - x[4] * np.cos(dt * x[4] + x[2])) / x[4] ** 2
        ).item(0)
        a25 = (
            (
                dt * x[5] * np.cos(dt * x[4] + x[2])
                - dt * (-dt * x[4] * x[5] - x[4] * x[3]) * np.sin(dt * x[4] + x[2])
                + x[3] * np.cos(x[2])
                + (-dt * x[5] - x[3]) * np.cos(dt * x[4] + x[2])
            )
            / x[4] ** 2
            - 2
            * (
                x[4] * x[3] * np.cos(x[2])
                - x[5] * np.sin(x[2])
                + x[5] * np.sin(dt * x[4] + x[2])
                + (-dt * x[4] * x[5] - x[4] * x[3]) * np.cos(dt * x[4] + x[2])
            )
            / x[4] ** 3
        ).item(0)
        a26 = (
            (
                -dt * x[4] * np.cos(dt * x[4] + x[2])
                - np.sin(x[2])
                + np.sin(dt * x[4] + x[2])
            )
            / x[4] ** 2
        ).item(0)
        a35 = self.dt
        a46 = self.dt
        F = np.array(
            [
                [1.0, 0.0, a13, a14, a15, a16],
                [0.0, 1.0, a23, a24, a25, a26],
                [0.0, 0.0, 1.0, 0.0, a35, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, a46],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F
    
    def predict(self):
        
        self.x = self.f(self.x)
        self.F = self.getF(self.x)
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        is_positive_semidefinite(self.P)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        return self.x    
        
    
class EKF_CA(KF_Base):
    """
    An EKF for tracking
    """

    def __init__(
        self, n=4, m=2, dt=None, P=None, Q=None, R=None, init_x=None, cfg=None
    ):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=cfg)

        self.dt = dt
        self.m = m
        self.JH = np.matrix(np.eye(self.m, 6))
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)

    def f(self, x):
        # State-transition function is identity

        # dt = dt if dt else self.dt
        x_fil = np.array(x)
        x_fil[0] = x[0] + self.dt * x[2] + 0.5 * self.dt * self.dt * x[4]
        x_fil[1] = x[1] + self.dt * x[3] + 0.5 * self.dt * self.dt * x[5]
        x_fil[2] = x[2] + self.dt * x[4]
        x_fil[3] = x[3] + self.dt * x[5]
        return x_fil

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        a13 = self.dt
        a14 = 0
        a23 = 0
        a24 = self.dt
        dt = self.dt
        at2 = self.dt * self.dt * 0.5
        F = np.matrix(
            [
                [1.0, 0.0, a13, a14, at2, 0],
                [0.0, 1.0, a23, a24, 0, at2],
                [0.0, 0.0, 1.0, 0.0, dt, 0],
                [0.0, 0.0, 0.0, 1.0, 0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F

    def h(self, x):
        # Observation function is identity
        return self.JH * x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH

    def update(self, z):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------
        self.x = self.f(self.x)
        self.F = self.getF(self.x)
        self.P = self.F * self.P * self.F.T + self.Q

        # Update -----------------------------------------------------
        G = self.P * self.H.T * np.linalg.inv(self.H * self.P * self.H.T + self.R)
        self.x += G * (np.array(z) - self.h(self.x).T).T
        #         self.P = np.matmul(self.I - np.matmul(G, self.H), self.P)
        self.P = (self.I - np.matmul(G, self.H)) * self.P
        # return self.x.asarray()
        return np.array(self.x.reshape(self.n))


class EKF_CTRA(KF_Base):
    """
    An EKF for tracking
    """

    def __init__(
        self, n=4, m=2, dt=None, P=None, Q=None, R=None, init_x=None, cfg=None
    ):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=cfg)

        self.dt = dt
        self.JH = np.matrix(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)

    def f(self, x):
        # State-transition function is identity
        # [x,y,yaw,v,w,a]

        x_fil = np.array(x)
        x_fil[0] = x[0] + (1 / x[4] ** 2) * (
            (x[3] * x[4] + x[5] * x[4] * self.dt) * np.sin(x[2] + x[4] * self.dt)
            + x[5] * np.cos(x[2] + x[4] * self.dt)
            - x[3] * x[4] * np.sin(x[2])
            - x[5] * np.cos(x[2])
        )
        x_fil[1] = x[1] + (1 / x[4] ** 2) * (
            (-x[3] * x[4] - x[5] * x[4] * self.dt) * np.cos(x[2] + x[4] * self.dt)
            + x[5] * np.sin(x[2] + x[4] * self.dt)
            + x[3] * x[4] * np.cos(x[2])
            - x[5] * np.sin(x[2])
        )
        x_fil[2] = x[2] + x[4] * self.dt
        x_fil[3] = x[3] + x[5] * self.dt
        return x_fil

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        dt = self.dt
        a13 = (
            (
                -x[4] * x[3] * np.cos(x[2])
                + x[5] * np.sin(x[2])
                - x[5] * np.sin(dt * x[4] + x[2])
                + (dt * x[4] * x[5] + x[4] * x[3]) * np.cos(dt * x[4] + x[2])
            )
            / x[4] ** 2
        ).item(0)

        a14 = (
            (-x[4] * np.sin(x[2]) + x[4] * np.sin(dt * x[4] + x[2])) / x[4] ** 2
        ).item(0)

        a15 = (
            (
                -dt * x[5] * np.sin(dt * x[4] + x[2])
                + dt * (dt * x[4] * x[5] + x[4] * x[3]) * np.cos(dt * x[4] + x[2])
                - x[3] * np.sin(x[2])
                + (dt * x[5] + x[3]) * np.sin(dt * x[4] + x[2])
            )
            / x[4] ** 2
            - 2
            * (
                -x[4] * x[3] * np.sin(x[2])
                - x[5] * np.cos(x[2])
                + x[5] * np.cos(dt * x[4] + x[2])
                + (dt * x[4] * x[5] + x[4] * x[3]) * np.sin(dt * x[4] + x[2])
            )
            / x[4] ** 3
        ).item(0)

        a16 = (
            (
                dt * x[4] * np.sin(dt * x[4] + x[2])
                - np.cos(x[2])
                + np.cos(dt * x[4] + x[2])
            )
            / x[4] ** 2
        ).item(0)

        a23 = (
            (
                -x[4] * x[3] * np.sin(x[2])
                - x[5] * np.cos(x[2])
                + x[5] * np.cos(dt * x[4] + x[2])
                - (-dt * x[4] * x[5] - x[4] * x[3]) * np.sin(dt * x[4] + x[2])
            )
            / x[4] ** 2
        ).item(0)
        a24 = (
            (x[4] * np.cos(x[2]) - x[4] * np.cos(dt * x[4] + x[2])) / x[4] ** 2
        ).item(0)
        a25 = (
            (
                dt * x[5] * np.cos(dt * x[4] + x[2])
                - dt * (-dt * x[4] * x[5] - x[4] * x[3]) * np.sin(dt * x[4] + x[2])
                + x[3] * np.cos(x[2])
                + (-dt * x[5] - x[3]) * np.cos(dt * x[4] + x[2])
            )
            / x[4] ** 2
            - 2
            * (
                x[4] * x[3] * np.cos(x[2])
                - x[5] * np.sin(x[2])
                + x[5] * np.sin(dt * x[4] + x[2])
                + (-dt * x[4] * x[5] - x[4] * x[3]) * np.cos(dt * x[4] + x[2])
            )
            / x[4] ** 3
        ).item(0)
        a26 = (
            (
                -dt * x[4] * np.cos(dt * x[4] + x[2])
                - np.sin(x[2])
                + np.sin(dt * x[4] + x[2])
            )
            / x[4] ** 2
        ).item(0)
        a35 = self.dt
        a46 = self.dt
        F = np.matrix(
            [
                [1.0, 0.0, a13, a14, a15, a16],
                [0.0, 1.0, a23, a24, a25, a26],
                [0.0, 0.0, 1.0, 0.0, a35, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, a46],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F

    def update(self, z):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------
        self.x = self.f(self.x)
        self.F = self.getF(self.x)
        self.P = self.F * self.P * self.F.T + self.Q

        # Update -----------------------------------------------------
        G = self.P * self.H.T * np.linalg.inv(self.H * self.P * self.H.T + self.R)
        self.x += G * (np.array(z) - self.h(self.x).T).T
        #         self.P = np.matmul(self.I - np.matmul(G, self.H), self.P)
        self.P = (self.I - np.matmul(G, self.H)) * self.P
        # return self.x.asarray()
        return np.array(self.x.reshape(self.n))

    def h(self, x):
        # Observation function is identity
        return self.JH * x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH


class EKF_RVBOX(KF_Base):
    """
    An EKF for tracking
    """

    def __init__(
        self, n=4, m=2, dt=None, P=None, Q=None, R=None, init_x=None, cfg=None
    ):
        KF_Base.__init__(self, n=n, m=m, P=P, Q=Q, R=R, init_x=init_x, cfg=cfg)

        self.dt = dt
        self.m = m
        self.JH = np.matrix(np.eye(self.m, 8))
        self.F = self.getF(self.x)
        self.H = self.getH(self.x)

    def f(self, x):
        # State-transition function is identity

        # dt = dt if dt else self.dt
        x_fil = np.array(x)
        x_fil[0] = x[0] + self.dt * x[4]
        x_fil[1] = x[1] + self.dt * x[5]
        x_fil[2] = x[2] + self.dt * x[6]
        x_fil[3] = x[3] + self.dt * x[7]
        return x_fil

    def getF(self, x):
        # So state-transition Jacobian is identity matrix
        F = np.matrix(
            [
                [1.0, 0.0, 0.0, 0.0, self.dt, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, self.dt, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, self.dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, self.dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        return F

    def h(self, x):
        # Observation function is identity
        return self.JH * x

    def getH(self, x):
        # So observation Jacobian is identity matrix
        return self.JH

    def update(self, z):
        """
        Runs one step of the EKF on observations z, where z is a tuple of length M.
        Returns a NumPy array representing the updated state.
        """
        # Predict ----------------------------------------------------
        self.x = self.f(self.x)
        self.F = self.getF(self.x)
        self.P = self.F * self.P * self.F.T + self.Q

        # Update -----------------------------------------------------
        G = self.P * self.H.T * np.linalg.inv(self.H * self.P * self.H.T + self.R)
        self.x += G * (np.array(z) - self.h(self.x).T).T
        #         self.P = np.matmul(self.I - np.matmul(G, self.H), self.P)
        self.P = (self.I - np.matmul(G, self.H)) * self.P
        # return self.x.asarray()
        return np.array(self.x.reshape(self.n))
