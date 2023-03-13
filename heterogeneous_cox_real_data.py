import autograd.numpy as np
import copy
from scipy.stats import multinomial
from scipy.stats import multivariate_normal
from scipy.stats import gamma
from scipy.stats import expon
from scipy.stats import uniform
from scipy.special import expit
from numpy.polynomial import legendre
from scipy.special import psi
from scipy.linalg import block_diag
from scipy.stats import poisson
from scipy.stats import bernoulli
from autograd import grad
from scipy.optimize import minimize
from tqdm import tqdm



class HGCox:
    """
    This class implements the inference for heterogeneous correlated Gaussian Cox process with auxiliary latent variables.
    The main feature is the mean-field variational inference. 
    """
    def __init__(self, number_of_tasks, number_of_latent, number_of_dimension):
        """
        Initialises an instance.

        :type number_of_tasks: numpy array
        :param number_of_tasks: number of tasks for point process, regression and classification
        :type number_of_latent: int
        :param number_of_latent: number of latent functions
        """
        self.number_of_tasks_pp = number_of_tasks[0]
        self.number_of_tasks_reg = number_of_tasks[1]
        self.number_of_tasks_cla = number_of_tasks[2]
        self.number_of_tasks = np.sum(number_of_tasks)
        self.number_of_latent = number_of_latent
        self.number_of_dimension = number_of_dimension

        # hyperparameters
        self.weight = np.zeros((self.number_of_tasks, self.number_of_latent))
        # self.weight_estimated = None
        self.weight_pp = np.zeros((self.number_of_tasks_pp, self.number_of_latent))
        self.weight_reg = np.zeros((self.number_of_tasks_reg, self.number_of_latent))
        self.weight_cla = np.zeros((self.number_of_tasks_cla, self.number_of_latent))
        self.theta = np.zeros((self.number_of_latent,2)) # 2 parameters for the RBF kernel
        self.noise_reg = np.zeros(self.number_of_tasks_reg)
        self.xm = None
        self.M = 0

        # parameters
        self.lamda_ub = np.zeros((self.number_of_tasks_pp,1))
        # self.lamda_ub_estimated = None

        # training and test data
        self.x_pp = None
        self.x_reg = None
        self.y_reg = None
        self.x_cla = None
        self.y_cla = None
        self.X = None

        self.x_pp_test = None
        self.x_reg_test = None
        self.y_reg_test = None
        self.x_cla_test = None
        self.y_cla_test = None
        self.X_test = None

    def set_hyperparameters(self, xm, weight, theta, noise_reg):
        r"""
        Fix the hyperparameters.
        Here we use the RBF kernel :math:`K(x,x')=\theta0*exp(-theta1/2*(||x - x'||**2))`. 
        The kernel for multi-output GP is :math:`K_{i,j}(x,x')=\sum_{z=1}^Z W[i][z]*W[j][z]*K_z(x,x')`. 
        
        :type xm: numpy array
        :param xm: the inducing points for all tasks 
        :type weight: numpy array
        :param weight: the mixing weight for all tasks
        :type theta: numpy array
        :param theta: the kernel hyperparameters for latent functions
        :type noise_reg: numpy array
        :param noise_reg: observation noise for regression tasks
        """
        assert xm.shape[-1]==self.number_of_dimension, 'dimension does not match'
        self.xm = xm
        self.M = len(xm)
        self.weight = weight
        self.weight_pp = self.weight[:self.number_of_tasks_pp]
        self.weight_reg = self.weight[self.number_of_tasks_pp:(self.number_of_tasks_pp+self.number_of_tasks_reg)]
        self.weight_cla = self.weight[(self.number_of_tasks_pp+self.number_of_tasks_reg):]
        self.theta = theta
        self.noise_reg = noise_reg

    def set_train_test_data(self, x_pp, x_reg, y_reg, x_cla, y_cla, X, x_pp_test, x_reg_test, y_reg_test, x_cla_test, y_cla_test, X_test):
        r"""
        Set the training/test data and observation window for point process, regression and classification tasks. 

        :type points_hawkes: list
        :param points_hawkes: the training points
        :type T: float
        :param T: the observation window [0,T] for training points
        :type points_hawkes_test: list
        :param points_hawkes_test: the test points
        :type T_test: float
        :param T_test: the observation window [0,T_test] for test points
        """
        assert len(x_pp)==self.number_of_tasks_pp, 'number of point process tasks does not match'
        assert len(x_reg)==len(y_reg)==self.number_of_tasks_reg, 'number of regression tasks does not match'
        assert len(x_cla)==len(y_cla)==self.number_of_tasks_cla, 'number of classification tasks does not match'
        assert x_pp[0].shape[-1]==x_reg[0].shape[-1]==x_cla[0].shape[-1]==\
        x_pp_test[0].shape[-1]==x_reg_test[0].shape[-1]==x_cla_test[0].shape[-1]==\
        len(X)==len(X_test)==self.number_of_dimension, 'dimension does not match'
        self.x_pp = x_pp
        self.x_reg = x_reg
        self.y_reg = y_reg
        self.x_cla = x_cla
        self.y_cla = y_cla
        self.X = X
        self.N_pp = np.array([len(x_pp[i]) for i in range(self.number_of_tasks_pp)]).reshape(-1,1)
        self.N_reg = np.array([len(x_reg[i]) for i in range(self.number_of_tasks_reg)]).reshape(-1,1)
        self.N_cla = np.array([len(x_cla[i]) for i in range(self.number_of_tasks_cla)]).reshape(-1,1)
        self.N = np.concatenate((self.N_pp,self.N_reg,self.N_cla))

        self.x_pp_test = x_pp_test
        self.x_reg_test = x_reg_test
        self.y_reg_test = y_reg_test
        self.x_cla_test = x_cla_test
        self.y_cla_test = y_cla_test
        self.X_test = X_test

#########################################################################################################

    'tool function'
    @staticmethod
    def loglikelihood_pp(points_inhomo, intensity, X): # points_inhomo is list of arrays (vector), intensity is T*M1*M2 matrix
        r"""
        Compute the loglikelihood of observation using given :math:`\mu` and :math:`\phi`.

        :type points_inhomo: list
        :param points_inhomo: the observed timestamps
        :type mu: float
        :param mu: the baseline intensity
        :type phi: numpy array
        :param phi: the influence function
        :type T_phi: float
        :param T_phi: the support of influence function
        :type T: float
        :param T: the observation window
        :rtype: float
        :return: the loglikelihood
        """
        assert intensity[0].ndim==len(X), 'dimension of intensity and X does not match'
        delta_x = X/intensity.shape[1:]
        logl=0
        for i in range(len(intensity)):
            logl+=np.sum(np.log(intensity[i][tuple((points_inhomo[i]/delta_x).astype(int).T)]))
            logl-=np.sum(intensity[i]*np.prod(delta_x))
        return logl

    @staticmethod
    def loglikelihood_pp_miss(points_inhomo, intensity, X, x_y_min, x_y_max): # points_inhomo is list of arrays (vector), intensity is T*M1*M2 matrix
        r"""
        Compute the loglikelihood of observation using given :math:`\mu` and :math:`\phi`.

        :type points_inhomo: list
        :param points_inhomo: the observed timestamps
        :type mu: float
        :param mu: the baseline intensity
        :type phi: numpy array
        :param phi: the influence function
        :type T_phi: float
        :param T_phi: the support of influence function
        :type T: float
        :param T: the observation window
        :rtype: float
        :return: the loglikelihood
        """
        assert intensity[0].ndim==len(X), 'dimension of intensity and X does not match'
        delta_x = X/intensity.shape[1:]
        res=[]
        for i in range(len(intensity)):
            index_min=(x_y_min[i]/delta_x).astype(int)
            index_max=(x_y_max[i]/delta_x).astype(int)
            logl=0
            logl+=np.sum(np.log(intensity[i][tuple((points_inhomo[i]/delta_x).astype(int).T)]))
            logl-=np.sum(intensity[i][index_min[0]:index_max[0],index_min[1]:index_max[1]]*np.prod(delta_x))
            res.append(logl)
        return res

    @staticmethod
    def loglikelihood_reg(x_reg, y_reg, f_reg, noise_reg, X): # x_reg and y_reg are lists of arrays (vector), f_reg is T*M1*M2 matrix
        r"""
        Compute the loglikelihood of observation using given :math:`\mu` and :math:`\phi`.

        :type x_reg: list of arrays
        :param x_reg: the input points for all regression tasks 
        :type y_reg: list of array
        :param y_reg: the output value for all regression tasks 
        :type f_reg: numpy array
        :param f_reg: the latent function values for all regression tasks
        :type noise_reg: numpy array
        :param noise_reg: the observation noise for all regression tasks
        :type X: numpy array
        :param X: the observation window
        :rtype: float
        :return: the loglikelihood
        """
        assert f_reg[0].ndim==len(X), 'dimension of intensity and X does not match'
        delta_x = X/f_reg.shape[1:]
        logl=0
        for i in range(len(f_reg)):
            logl+=np.sum(np.log(multivariate_normal(f_reg[i][tuple((x_reg[i]/delta_x).astype(int).T)], np.eye(len(x_reg[i]))*noise_reg[i]).pdf(y_reg[i])))
        return logl

    @staticmethod
    def loglikelihood_cla(x_cla, y_cla, f_cla, X): # x_cla and y_cla are lists of arrays (vector), f_cla is T*M1*M2 matrix
        r"""
        Compute the loglikelihood of observation using given :math:`\mu` and :math:`\phi`.

        :type x_cla: list of arrays
        :param x_cla: the input points for all classification tasks 
        :type y_cla: list of array
        :param y_cla: the output value for all classification tasks 
        :type f_cla: numpy array
        :param f_cla: the latent function values for all classification tasks
        :type X: numpy array
        :param X: the observation window
        :rtype: float
        :return: the loglikelihood
        """
        assert f_cla[0].ndim==len(X), 'dimension of intensity and X does not match'
        delta_x = X/f_cla.shape[1:]
        logl=0
        for i in range(len(f_cla)):
            logl+=np.sum(np.log(expit(f_cla[i][tuple((x_cla[i]/delta_x).astype(int).T)]*y_cla[i])))
        return logl

    @staticmethod
    def rbf_kernel(theta, x1, x2):
        r"""
        Compute the kernel matrix at x1, x2, :math:`K(x1,x2)=\theta0*exp(-theta1/2*(||x1 - x2||**2))`. 

        :type theta: numpy array
        :param theta: kernel hyperparameters, [theta0, theta1] 
        :type x1: numpy array
        :param x1: the location points
        :type x2: numpy array
        :param x2: the location points

        :rtype: 2D numpy array
        :return: kernel matrix
        """
        assert x1.shape[1]==x2.shape[1],'dimension does not match'
        theta0 = theta[0]
        theta1 = theta[1]
        N1=x1.shape[0]
        N2=x2.shape[0]
        D=x1.shape[1]
        a=x1.reshape(N1,1,D)
        b=x2.reshape(1,N2,D)
        return theta0*np.exp(-theta1/2*np.sum((a-b)**2,axis=2))

    def rbf_kernel_mo(self, W, theta, x1, x2 = None): 
        r"""
        Compute the kernel matrix at x1, x2 for i-th task (if W is a vector W[i]) and for all tasks (if W is a matrix) :math:`K_z(x1,x2)=\theta0[z]*exp(-theta1[z]/2*(||x1 - x2||**2))`. 
        The kernel for i-th task is :math:`K_i(x1, x2)=\sum_{z=1}^Z w[i][z]*w[i][z]*K_z(x1, x2)`. 

        :type W: numpy array
        :param W: the mixing weight for all tasks
        :type theta: numpy array
        :param theta: the kernel hyperparameters for latent functions
        :type x1: numpy array
        :param x1: the location points
        :type x2: numpy array
        :param x2: the location points

        :rtype: 2D numpy array
        :return: kernel matrix
        """
        if W.ndim==2: # W is a matrix
            assert W.shape[1]==len(theta), 'number of latent functions does not match'
            T = W.shape[0] # number of tasks
            Z = W.shape[1] # number of latent functions
            if x2 is None:
                M = len(x1) # number of inducing points
                N = M*T # number of all inducing points
                K=np.zeros((N,N))
                for z in range(Z):
                    K+=np.kron(np.outer(W[:,z],W[:,z]),self.rbf_kernel(theta[z],x1,x1))
                return K+np.eye(N)*1e-5
            else:
                M1 = len(x1) # number of x1 on each task
                N1 = M1*T
                M2 = len(x2) # number of x2 on each task
                N2 = M2*T
                K=np.zeros((N1,N2))
                for z in range(Z):
                    K+=np.kron(np.outer(W[:,z],W[:,z]),self.rbf_kernel(theta[z],x1,x2))
                return K
        elif W.ndim==1: # W is W[i]
            assert len(W)==len(theta), 'number of latent functions does not match'
            Z=len(W) # number of latent functions
            if x2 is None:
                N_1=len(x1) # number of points for x1
                K=np.zeros((N_1,N_1))
                for z in range(Z):
                    K+=W[z]**2*self.rbf_kernel(theta[z],x1,x1)
                return K+np.eye(N_1)*1e-5
            else:
                N_1=len(x1) # number of points for x1
                N_2=len(x2) # number of points for x2
                K=np.zeros((N_1,N_2))
                for z in range(Z):
                    K+=W[z]**2*self.rbf_kernel(theta[z],x1,x2)
                return K
        else:
            raise TypeError('The type of W is wrong')

#########################################################################################################
    'simulation'
    @staticmethod
    def simulation(number_of_tasks, intensity, f_reg, x_reg, noise_reg, p_cla, x_cla, X): # discrete f that is a matrix T*M1*M2, x_reg and x_cla are matrix T*M1*M2*D
        r"""
        simulate point process data, regression data and classification data given number of each kind of tasks, latent functions f, intensity upperbound and support.

        :type number_of_tasks: numpy array
        :param number_of_tasks: the number of each kind of tasks, [pp, reg, cla]
        :type f: numpy array
        :param f: the latent functions for each task
        :type lamda_ub: numpy array
        :param lamda_ub: the intensity upper bound for each point process task
        :type X: numpy array
        :param X: the support on each dimension

        :return: simulated data
        """
        assert x_reg[0].shape[-1] == len(X), 'dimension does not match'
        assert intensity.ndim == f_reg.ndim == p_cla.ndim, 'dimension does not match'
        # assert f.shape[0] == np.sum(number_of_tasks), 'number of tasks of f does not match'
        D = len(X)
        delta_x = X/intensity.shape[1:]
        # intensity = lamda_ub * expit(f[:number_of_tasks[0]])
        # f_reg = f[number_of_tasks[0]:number_of_tasks[0]+number_of_tasks[1]]
        # p_cla = expit(f[number_of_tasks[0]+number_of_tasks[1]:])

        x_pp=[]
        y_reg=[]
        y_cla=[]

        for i in range(number_of_tasks[0]):
            Lambda = np.max(intensity[i])*np.prod(X)
            r = poisson.rvs(Lambda)
            x = np.array([uniform.rvs(size=r)*X[j] for j in range(D)]).T
            measures = intensity[i][tuple((x/delta_x).astype(int).T)]
            ar = measures/np.max(intensity[i])
            index = bernoulli.rvs(ar)
            points = x[index.astype(bool)]
            x_pp.append(points)

        for i in range(number_of_tasks[1]):
            x_reg_i = x_reg[i].reshape(-1, D)
            mean_reg_i = f_reg[i][tuple((x_reg_i/delta_x).astype(int).T)]
            y_reg_i = multivariate_normal(mean_reg_i, np.eye(len(mean_reg_i))*noise_reg[i]).rvs()
            y_reg.append(y_reg_i)

        for i in range(number_of_tasks[2]):
            x_cla_i = x_cla[i].reshape(-1, D)
            p_i = p_cla[i][tuple((x_cla_i/delta_x).astype(int).T)]
            y_cla_i = bernoulli(p_i).rvs()
            y_cla_i[y_cla_i==0] = -1
            y_cla.append(y_cla_i)
        return x_pp, y_reg, y_cla

    @staticmethod
    def simulation_f(number_of_tasks, f, lamda_ub, x_reg, noise_reg, x_cla, X): # discrete f that is a matrix T*M1*M2, x_reg and x_cla are matrix T*M1*M2*D
        r"""
        simulate point process data, regression data and classification data given number of each kind of tasks, latent functions f, intensity upperbound and support.

        :type number_of_tasks: numpy array
        :param number_of_tasks: the number of each kind of tasks, [pp, reg, cla]
        :type f: numpy array
        :param f: the latent functions for each task
        :type lamda_ub: numpy array
        :param lamda_ub: the intensity upper bound for each point process task
        :type x_reg: list of numpy array
        :param x_reg: the input points for regression tasks
        :type noise_reg: numpy array
        :param noise_reg: the observation noise for regression tasks
        :type x_cla: list of numpy array
        :param x_cla: the input points for classification tasks
        :type X: numpy array
        :param X: the support on each dimension

        :return: simulated data
        """
        assert x_reg[0].shape[-1] == len(X), 'dimension does not match'
        # assert f.ndim == 3, 'dimension of f does not match'
        assert f.shape[0] == np.sum(number_of_tasks), 'number of tasks of f does not match'
        D = len(X)
        delta_x = X/f.shape[1:]
        intensity = lamda_ub * expit(f[:number_of_tasks[0]])
        f_reg = f[number_of_tasks[0]:number_of_tasks[0]+number_of_tasks[1]]
        p_cla = expit(f[number_of_tasks[0]+number_of_tasks[1]:])

        x_pp=[]
        y_reg=[]
        y_cla=[]

        for i in range(number_of_tasks[0]):
            Lambda = lamda_ub[i]*np.prod(X)
            r = poisson.rvs(Lambda)
            x = np.array([uniform.rvs(size=r)*X[j] for j in range(D)]).T
            measures = intensity[i][tuple((x/delta_x).astype(int).T)]
            ar = measures/lamda_ub[i]
            index = bernoulli.rvs(ar)
            points = x[index.astype(bool)]
            x_pp.append(points)

        for i in range(number_of_tasks[1]):
            x_reg_i = x_reg[i].reshape(-1, D)
            mean_reg_i = f_reg[i][tuple((x_reg_i/delta_x).astype(int).T)]
            y_reg_i = multivariate_normal(mean_reg_i, np.eye(len(mean_reg_i))*noise_reg[i]).rvs()
            y_reg.append(y_reg_i)

        for i in range(number_of_tasks[2]):
            x_cla_i = x_cla[i].reshape(-1, D)
            p_i = p_cla[i][tuple((x_cla_i/delta_x).astype(int).T)]
            y_cla_i = bernoulli(p_i).rvs()
            y_cla_i[y_cla_i==0] = -1
            y_cla.append(y_cla_i)
        return x_pp, y_reg, y_cla

#########################################################################################################


    'Mean-Field Variational Inference'
    @staticmethod
    def gq_points_weights(a,b,Q):
        r"""
        Generate the Gaussian quadrature nodes and weights for the integral :math:`\int_a^b f(t) dt`

        :type a: float
        :param a: the lower end of the integral
        :type b: float
        :param b: the upper end of the integral
        :type Q: int
        :param Q: the number of Gaussian quadrature nodes (weights)
        :rtype: 1D numpy array, 1D numpy array
        :return: Gaussian quadrature nodes and the corresponding weights
        """
        p,w = legendre.leggauss(Q)
        c = np.array([0] * Q + [1])
        p_new = (a + b + (b - a) * p) / 2
        w_new = (b - a) / (legendre.legval(p, legendre.legder(c))**2*(1-p**2))
        return p_new.reshape(-1,1),w_new.reshape(-1,1)

    def a_c_predict(self, ym_mean, ym_cov, W, theta, K_MM_inv, x_pre): 
        r"""
        The mean, sqrt(E[y_pred^2]) and covariance of y_pred based on x_M, y_M (Gaussian process regression). 

        :type ym_mean: numpy array
        :param ym_mean: input y mean
        :type ym_cov: 2D numpy array
        :param ym_cov: input y covariance
        :type W: numpy array
        :param W: mixing weight, 1D for specific task and 2D for all tasks
        :type theta: numpy array
        :param theta: hyperparameters for kernel
        :type K_MM_inv: 2D numpy array
        :param K_MM_inv: the inverse of kernel matrix of xm
        :type x_pre: numpy array
        :param x_pre: the predictive points

        :rtype: numpy arrays
        :return: mean, sqrt(E[y_pred^2]) and covariance of y_pred
        """
        k = self.rbf_kernel_mo(W, theta, x_pre, self.xm)
        k_C = np.dot(k, K_MM_inv)
        y_pred_mean = np.dot(k_C, ym_mean)
        k_matrix_pre = self.rbf_kernel_mo(W, theta, x_pre)
        y_pred_cov = k_matrix_pre - np.dot(k_C,k.T) + np.dot(np.dot(k_C, ym_cov), k_C.T)
    #     y_pred_cov = np.dot(np.dot(k_C, ym_cov), k_C.T)
        return y_pred_mean, np.sqrt(np.diag(y_pred_cov)+y_pred_mean**2), y_pred_cov

    def ELBO(self, hp, mean_fm, cov_fm):
        W = hp[:(self.number_of_tasks*self.number_of_latent)].reshape((self.number_of_tasks, self.number_of_latent))
        theta = hp[(self.number_of_tasks*self.number_of_latent):].reshape((self.number_of_latent, 2))
        K_MM = self.rbf_kernel_mo(W, theta, self.xm)
        K_MM_inv = np.linalg.inv(K_MM)
        L = np.linalg.slogdet(K_MM)[1] + np.trace(np.dot(K_MM_inv,cov_fm)) + np.dot(np.dot(mean_fm,K_MM_inv),mean_fm)
        return L

    def MF(self, num_gq, num_pre, num_iter, gap_min, gap_max, hp_tune=False):
        r"""
        Mean-field variational inference algorithm which is used to estimate the posterior of lamda_ub, f^p(x_s), f^r(x_s) and f^c(x_s). 

        :type num_gq: numpy array
        :param num_gq: the number of Gaussian quadrature nodes on [0,X_1],...,[0,X_D]
        :type num_pre: numpy array
        :param num_pre: the number of prediction points on [0,X_1],...,[0,X_D]
        :type num_iter: int
        :param num_iter: the number of MF iterations

        :rtype: numpy array
        :return: the mean and covariance of [f^p(x_s), f^r(x_s), f^c(x_s)], the location parameter of lamda_ub, and the training and 
        test log-likelihood along MF iterations. 
        """
        if self.number_of_dimension==1:
            x_pre = np.linspace(0,self.X[0],num_pre[0]).reshape(-1,1)
        elif self.number_of_dimension==2:
            x_temp, y_temp = np.meshgrid(*[np.linspace(0,self.X[i],num_pre[i]).reshape(-1,1) for i in range(self.number_of_dimension)])
            x_pre = np.vstack([x_temp.ravel('F'), y_temp.ravel('F')]).T # 2D simulation
            assert x_pre.shape[-1]==self.number_of_dimension, 'dimension of input does not match'
        else:
            raise TypeError('Only accept 1D/2D inputs')

        K_MM = self.rbf_kernel_mo(self.weight, self.theta, self.xm) # K_MM for all tasks
        K_MM_inv = np.linalg.inv(K_MM)

        K_MM_pp = np.zeros((self.number_of_tasks_pp, self.M, self.M)) # K_MM for point process tasks
        K_MM_pp_inv = np.zeros((self.number_of_tasks_pp, self.M, self.M))
        for i in range(self.number_of_tasks_pp):
            K_MM_pp[i] = self.rbf_kernel_mo(self.weight_pp[i], self.theta, self.xm)
            K_MM_pp_inv[i] = np.linalg.inv(K_MM_pp[i])

        K_MM_reg = np.zeros((self.number_of_tasks_reg, self.M, self.M)) # K_MM for regression tasks
        K_MM_reg_inv = np.zeros((self.number_of_tasks_reg, self.M, self.M))
        for i in range(self.number_of_tasks_reg):
            K_MM_reg[i] = self.rbf_kernel_mo(self.weight_reg[i], self.theta, self.xm)
            K_MM_reg_inv[i] = np.linalg.inv(K_MM_reg[i])

        K_MM_cla = np.zeros((self.number_of_tasks_cla, self.M, self.M)) # K_MM for classification tasks
        K_MM_cla_inv = np.zeros((self.number_of_tasks_cla, self.M, self.M))
        for i in range(self.number_of_tasks_cla):
            K_MM_cla[i] = self.rbf_kernel_mo(self.weight_cla[i], self.theta, self.xm)
            K_MM_cla_inv[i] = np.linalg.inv(K_MM_cla[i])
        
        K_NM_pp = [np.zeros((self.N_pp[i][0], self.M)) for i in range(self.number_of_tasks_pp)] # K_NM for point process tasks
        for i in range(self.number_of_tasks_pp):
            K_NM_pp[i] = self.rbf_kernel_mo(self.weight_pp[i], self.theta, self.x_pp[i], self.xm)

        K_NM_reg = [np.zeros((self.N_reg[i][0], self.M)) for i in range(self.number_of_tasks_reg)] # K_NM for regression tasks
        for i in range(self.number_of_tasks_reg):
            K_NM_reg[i] = self.rbf_kernel_mo(self.weight_reg[i], self.theta, self.x_reg[i], self.xm)

        K_NM_cla = [np.zeros((self.N_cla[i][0], self.M)) for i in range(self.number_of_tasks_cla)] # K_NM for classification tasks
        for i in range(self.number_of_tasks_cla):
            K_NM_cla[i] = self.rbf_kernel_mo(self.weight_cla[i], self.theta, self.x_cla[i], self.xm)

        K_inv_K_reg = [np.dot(K_MM_reg_inv[i], K_NM_reg[i].T) for i in range(self.number_of_tasks_reg)]
        K_inv_K_cla = [np.dot(K_MM_cla_inv[i], K_NM_cla[i].T) for i in range(self.number_of_tasks_cla)]

        # 1D/2D Gaussian quadreture points and weights for all tasks
        if self.number_of_dimension==1:
            p_gq, w_gq = self.gq_points_weights(0, self.X[0], num_gq[0])
        elif self.number_of_dimension==2:
            p_gq_0, w_gq_0 = self.gq_points_weights(0, self.X[0], num_gq[0])
            p_gq_1, w_gq_1 = self.gq_points_weights(0, self.X[1], num_gq[1])
            w_gq = np.outer(w_gq_0,w_gq_1).ravel()
            temp_x, temp_y = np.meshgrid(p_gq_0,p_gq_1)
            p_gq = np.vstack([temp_x.ravel('F'), temp_y.ravel('F')]).T
        else:
            raise TypeError('Only accept 1D/2D inputs')
        
        K_gqM_pp = np.zeros((self.number_of_tasks_pp, len(p_gq), self.M)) # K_gqM for point process tasks
        for i in range(self.number_of_tasks_pp):
            K_gqM_pp[i] = self.rbf_kernel_mo(self.weight_pp[i], self.theta, p_gq, self.xm)
        
        # initial q_fm and q_lamda_ub
        # q_lamda_ub is a gamma distribution p_ga(alpha=N_pp_i+E(|Pi_i|),scale=1/X)
        alpha = self.N_pp*np.random.uniform(1,2)
        # q_fm is a Gaussian distribution N(mean_fm,cov_fm)
        mean_fm = np.random.uniform(-1, 1, size = self.M*self.number_of_tasks)
        cov_fm = K_MM
        

        # take out the mean and covariance for each task
        mean_fm_pp = [mean_fm[i*self.M:(i+1)*self.M] for i in range(self.number_of_tasks_pp)]
        mean_fm_reg = [mean_fm[(self.number_of_tasks_pp+i)*self.M:(self.number_of_tasks_pp+i+1)*self.M] for i in range(self.number_of_tasks_reg)]
        mean_fm_cla = [mean_fm[(self.number_of_tasks_pp+self.number_of_tasks_reg+i)*self.M:(self.number_of_tasks_pp+self.number_of_tasks_reg+i+1)*self.M] for i in range(self.number_of_tasks_cla)]

        cov_fm_pp = [cov_fm[i*self.M:(i+1)*self.M,i*self.M:(i+1)*self.M] for i in range(self.number_of_tasks_pp)]
        cov_fm_reg = [cov_fm[(self.number_of_tasks_pp+i)*self.M:(self.number_of_tasks_pp+i+1)*self.M,\
        (self.number_of_tasks_pp+i)*self.M:(self.number_of_tasks_pp+i+1)*self.M] for i in range(self.number_of_tasks_reg)]
        cov_fm_cla = [cov_fm[(self.number_of_tasks_pp+self.number_of_tasks_reg+i)*self.M:(self.number_of_tasks_pp+self.number_of_tasks_reg+i+1)*self.M,\
        (self.number_of_tasks_pp+self.number_of_tasks_reg+i)*self.M:(self.number_of_tasks_pp+self.number_of_tasks_reg+i+1)*self.M] for i in range(self.number_of_tasks_cla)]

        E_w_pp = [np.zeros(len(self.x_pp[i])) for i in range(self.number_of_tasks_pp)]
        a_gq_pp = np.zeros((self.number_of_tasks_pp,len(p_gq)))
        c_gq_pp = np.zeros((self.number_of_tasks_pp,len(p_gq)))
        int_intensity = np.zeros(self.number_of_tasks_pp)
        E_w_cla = [np.zeros(len(self.x_cla[i])) for i in range(self.number_of_tasks_cla)]
        
        if hp_tune == True:
            gradient_hp = grad(self.ELBO)
        # mean_fm_list = []
        # cov_fm_list = []
        # alpha_list = []
        logl_train_pp_list = []
        logl_test_pp_list = []
        logl_train_reg_list = []
        logl_test_reg_list = []
        logl_train_cla_list = []
        logl_test_cla_list = []
        hp_list = []
        
        for iteration in tqdm(range(num_iter), position = 0, leave = True):
            # update parameters of density of q_w for classification tasks: PG(w^c_(i,n)|1,c^c_{i,n})
            for i in range(self.number_of_tasks_cla):
                a, c, _ = self.a_c_predict(mean_fm_cla[i], cov_fm_cla[i], self.weight_cla[i], self.theta, K_MM_cla_inv[i], self.x_cla[i])
                E_w_cla[i] = 1/2/c*np.tanh(c/2)

            # update parameters of density of q_w for point process tasks: PG(w^p_(i,n)|1,c^p_{i,n})
            for i in range(self.number_of_tasks_pp):
                a, c, _ = self.a_c_predict(mean_fm_pp[i], cov_fm_pp[i], self.weight_pp[i], self.theta, K_MM_pp_inv[i], self.x_pp[i])
                E_w_pp[i] = 1/2/c*np.tanh(c/2)

            # update parameters of q_pi intensity=exp(E(log lamda))sigmoid(-c(t))exp((c(t)-a(t))/2)*P_pg(w|1,c(t))
            lamda_1 = np.exp(np.log(1/np.prod(self.X))+psi(alpha))

            # update parameters of q_lamda q_lamda=gamma(alpha=N^p_i+E(|pi|),scale=1/X_0/X_1)
            for i in range(self.number_of_tasks_pp):
                a_gq_pp[i], c_gq_pp[i], _ = self.a_c_predict(mean_fm_pp[i], cov_fm_pp[i], self.weight_pp[i], self.theta, K_MM_pp_inv[i], p_gq)
            int_intensity = (lamda_1*expit(-c_gq_pp)*np.exp((c_gq_pp-a_gq_pp)/2)).dot(w_gq)
            alpha = self.N_pp + int_intensity.reshape((-1,1))
            
            # update parameters of q_fm  q_fm=N(mean_fm,cov_fm)
            int_A=np.zeros((self.number_of_tasks_pp, self.M, self.M))
            for i in range(self.number_of_tasks_pp):
                int_A[i]+=E_w_pp[i].reshape(1,-1).dot((K_NM_pp[i].reshape(-1,self.M,1)*K_NM_pp[i].reshape(-1,1,self.M)).reshape(self.N_pp[i][0],-1)).reshape(self.M,self.M)
                int_A[i]+=((lamda_1[i]/2/c_gq_pp[i]*np.tanh(c_gq_pp[i]/2)*expit(-c_gq_pp[i])*np.exp((c_gq_pp[i]-a_gq_pp[i])/2)).reshape(-1,1)*\
                 (K_gqM_pp[i].reshape(-1,self.M,1)*K_gqM_pp[i].reshape(-1,1,self.M)).reshape(len(p_gq),-1)).T.dot(w_gq).reshape(self.M,self.M)
            int_B=np.zeros((self.number_of_tasks_pp,self.M))
            for i in range(self.number_of_tasks_pp):
                int_B[i]+=(0.5*np.ones(self.N_pp[i][0])).reshape(1,-1).dot(K_NM_pp[i]).reshape(self.M)
                int_B[i]+=((-1/2*lamda_1[i]*expit(-c_gq_pp[i])*np.exp((c_gq_pp[i]-a_gq_pp[i])/2)).reshape(-1,1)*K_gqM_pp[i]).T.dot(w_gq).reshape(self.M)

            H_pp=[K_MM_pp_inv[i].dot(int_A[i]).dot(K_MM_pp_inv[i]) for i in range(self.number_of_tasks_pp)]
            H_reg=[K_inv_K_reg[i].dot(np.eye(self.N_reg[i][0])/self.noise_reg[i]).dot(K_inv_K_reg[i].T) for i in range(self.number_of_tasks_reg)]
            H_cla=[K_inv_K_cla[i].dot(np.diag(E_w_cla[i])).dot(K_inv_K_cla[i].T) for i in range(self.number_of_tasks_cla)]
            H=block_diag(*(H_pp + H_reg + H_cla))
            v_pp = [K_MM_pp_inv[i].dot(int_B[i]) for i in range(self.number_of_tasks_pp)]
            v_reg= [K_inv_K_reg[i].dot(self.y_reg[i].reshape(-1)/self.noise_reg[i]) for i in range(self.number_of_tasks_reg)]
            v_cla= [K_inv_K_cla[i].dot(self.y_cla[i].reshape(-1)/2) for i in range(self.number_of_tasks_cla)]
            v=np.concatenate(v_pp+v_reg+v_cla)

            cov_fm = np.linalg.inv(H+K_MM_inv)
            mean_fm = cov_fm.dot(v)

            mean_fm_pp = mean_fm[:self.M*self.number_of_tasks_pp].reshape(self.number_of_tasks_pp,self.M)
            mean_fm_reg = mean_fm[self.M*self.number_of_tasks_pp:self.M*(self.number_of_tasks_pp+self.number_of_tasks_reg)].reshape(self.number_of_tasks_reg,self.M)
            mean_fm_cla = mean_fm[self.M*(self.number_of_tasks_pp+self.number_of_tasks_reg):].reshape(self.number_of_tasks_cla,self.M)

            cov_fm_pp = [cov_fm[i*self.M:(i+1)*self.M,i*self.M:(i+1)*self.M] for i in range(self.number_of_tasks_pp)]
            cov_fm_reg = [cov_fm[(self.number_of_tasks_pp+i)*self.M:(self.number_of_tasks_pp+i+1)*self.M,(self.number_of_tasks_pp+i)*self.M:(self.number_of_tasks_pp+i+1)*self.M] \
                          for i in range(self.number_of_tasks_reg)]
            cov_fm_cla = [cov_fm[(self.number_of_tasks_pp+self.number_of_tasks_reg+i)*self.M:(self.number_of_tasks_pp+self.number_of_tasks_reg+i+1)*self.M,\
                          (self.number_of_tasks_pp+self.number_of_tasks_reg+i)*self.M:(self.number_of_tasks_pp+self.number_of_tasks_reg+i+1)*self.M] for i in range(self.number_of_tasks_cla)]

            # update hyperparameters
            if hp_tune == True and (iteration+1)%2 == 0: # update hyperparameters every two iterations
                initial_guss = np.append(self.weight.ravel(),self.theta.ravel())
                res = minimize(self.ELBO, initial_guss, args=(mean_fm,cov_fm), jac = gradient_hp, method='SLSQP', \
                    bounds=[(1e-4,None)]*((self.number_of_tasks+2)*self.number_of_latent), options={'maxiter': 1})
                self.weight = res.x[:self.number_of_tasks*self.number_of_latent].reshape(self.number_of_tasks,self.number_of_latent)
                self.theta = res.x[self.number_of_tasks*self.number_of_latent:].reshape(self.number_of_latent,2)
                self.weight_pp = self.weight[:self.number_of_tasks_pp]
                self.weight_reg = self.weight[self.number_of_tasks_pp:(self.number_of_tasks_pp+self.number_of_tasks_reg)]
                self.weight_cla = self.weight[(self.number_of_tasks_pp+self.number_of_tasks_reg):]

                # update all kernel matrixes and inversion
                K_MM = self.rbf_kernel_mo(self.weight, self.theta, self.xm) # K_MM for all tasks
                K_MM_inv = np.linalg.inv(K_MM)

                for i in range(self.number_of_tasks_pp):
                    K_MM_pp[i] = self.rbf_kernel_mo(self.weight_pp[i], self.theta, self.xm)
                    K_MM_pp_inv[i] = np.linalg.inv(K_MM_pp[i])

                for i in range(self.number_of_tasks_reg):
                    K_MM_reg[i] = self.rbf_kernel_mo(self.weight_reg[i], self.theta, self.xm)
                    K_MM_reg_inv[i] = np.linalg.inv(K_MM_reg[i])

                for i in range(self.number_of_tasks_cla):
                    K_MM_cla[i] = self.rbf_kernel_mo(self.weight_cla[i], self.theta, self.xm)
                    K_MM_cla_inv[i] = np.linalg.inv(K_MM_cla[i])
                
                for i in range(self.number_of_tasks_pp):
                    K_NM_pp[i] = self.rbf_kernel_mo(self.weight_pp[i], self.theta, self.x_pp[i], self.xm)

                for i in range(self.number_of_tasks_reg):
                    K_NM_reg[i] = self.rbf_kernel_mo(self.weight_reg[i], self.theta, self.x_reg[i], self.xm)

                for i in range(self.number_of_tasks_cla):
                    K_NM_cla[i] = self.rbf_kernel_mo(self.weight_cla[i], self.theta, self.x_cla[i], self.xm)

                K_inv_K_reg = [np.dot(K_MM_reg_inv[i], K_NM_reg[i].T) for i in range(self.number_of_tasks_reg)]
                K_inv_K_cla = [np.dot(K_MM_cla_inv[i], K_NM_cla[i].T) for i in range(self.number_of_tasks_cla)]

                for i in range(self.number_of_tasks_pp):
                    K_gqM_pp[i] = self.rbf_kernel_mo(self.weight_pp[i], self.theta, p_gq, self.xm)

                # update noise variance 
                for i in range(self.number_of_tasks_reg):
                    a, c, _ = self.a_c_predict(mean_fm_reg[i], cov_fm_reg[i], self.weight_reg[i], self.theta, K_MM_reg_inv[i], self.x_reg[i])
                    self.noise_reg[i] = np.sum(self.y_reg[i].ravel()**2-2*self.y_reg[i].ravel()*a+c**2)/self.N_reg[i]

                hp_list.append(np.append(res.x,self.noise_reg))

            # compute the mean and covariance of g_{\mu}(t) and g_{\phi}(\tau) on finer grid and the corresponding train/test loglikelihood using mean
            mean_f_pre, _, cov_f_pre = self.a_c_predict(mean_fm, cov_fm, self.weight, self.theta, K_MM_inv, x_pre)
            mean_f_pre_pp = mean_f_pre[:np.prod(num_pre)*self.number_of_tasks_pp].reshape(self.number_of_tasks_pp,np.prod(num_pre))
            mean_f_pre_reg = mean_f_pre[np.prod(num_pre)*self.number_of_tasks_pp:np.prod(num_pre)*(self.number_of_tasks_pp+self.number_of_tasks_reg)].reshape(self.number_of_tasks_reg,np.prod(num_pre))
            mean_f_pre_cla = mean_f_pre[np.prod(num_pre)*(self.number_of_tasks_pp+self.number_of_tasks_reg):].reshape(self.number_of_tasks_cla,np.prod(num_pre))


            mean_lamda_ub_pre = alpha/np.prod(self.X)
            mean_intensity_pre = mean_lamda_ub_pre*expit(mean_f_pre_pp)
            logl_train_pp = self.loglikelihood_pp(self.x_pp, mean_intensity_pre.reshape(np.append(self.number_of_tasks_pp,num_pre)), self.X)
            # logl_test_pp = self.loglikelihood_pp(self.x_pp_test, mean_intensity_pre.reshape(np.append(self.number_of_tasks_pp,num_pre)), self.X_test)
            logl_test_pp = self.loglikelihood_pp_miss(self.x_pp_test, mean_intensity_pre.reshape(np.append(self.number_of_tasks_pp,num_pre)), self.X_test, gap_min, gap_max)
            logl_train_reg = self.loglikelihood_reg(self.x_reg, self.y_reg, mean_f_pre_reg.reshape(np.append(self.number_of_tasks_reg,num_pre)), self.noise_reg, self.X)
            logl_test_reg = self.loglikelihood_reg(self.x_reg_test, self.y_reg_test, mean_f_pre_reg.reshape(np.append(self.number_of_tasks_reg,num_pre)), self.noise_reg, self.X_test)
            logl_train_cla = self.loglikelihood_cla(self.x_cla, self.y_cla, mean_f_pre_cla.reshape(np.append(self.number_of_tasks_cla,num_pre)), self.X)
            logl_test_cla = self.loglikelihood_cla(self.x_cla_test, self.y_cla_test, mean_f_pre_cla.reshape(np.append(self.number_of_tasks_cla,num_pre)), self.X_test)

            # record
            # g_mu_mean_list.append(mean_g_mu)
            # g_mu_cov_list.append(cov_g_mu)
            # alpha_mu_list.append(alpha_mu)
            logl_train_pp_list.append(logl_train_pp)
            logl_test_pp_list.append(logl_test_pp)
            logl_train_reg_list.append(logl_train_reg)
            logl_test_reg_list.append(logl_test_reg)
            logl_train_cla_list.append(logl_train_cla)
            logl_test_cla_list.append(logl_test_cla)
        # mean_f_pre, _, cov_f_pre = self.a_c_predict(mean_fm, cov_fm, self.weight, self.theta, K_MM_inv, x_pre)
        return mean_f_pre, cov_f_pre, alpha, logl_train_pp_list, logl_test_pp_list, logl_train_reg_list, logl_test_reg_list, logl_train_cla_list, logl_test_cla_list, np.array(hp_list)