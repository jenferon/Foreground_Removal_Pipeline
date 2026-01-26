import numpy as np
from sklearn.decomposition import FastICA
from gpr4im import fg_tools as fg

def fastica(data, comps, shape):
    """
    Code to utilise the FastICA algorithm on 
    
    Inputs:
    data (numpy array): data to be fitted, assumed in the shape (npix,npix,nfreq)
    comps (int): number of components to fit to
    
    Outputs
    model (numpy array): model produced by FastICA algorithm in the shape (npix,npix,nfreq)
    residuals (numpy array): residuals produced by data - model in the shape (npix,npix,nfreq)
    """
    shape=data.shape
    f_ica = FastICA(n_components=comps)
    #generate the 4 components
    S = f_ica.fit_transform(data.reshape((shape[0]*shape[1],shape[2])))
    
    #get mixing matrix
    A = f_ica.mixing_
    
    #make model
    model_fICA = (np.matmul(A,S.T).T).reshape((shape[0],shape[1],shape[2]))
    
    #get resids
    resids_fICA = data - model_fICA #residuals 
    
    return model_fICA, resids_fICA

def GPR(data, frequency):
    """
    Code to utilise Gaussian Process Regression for numpy data cubes,
    with kernels as used in Mertens et al (2020)
    
    Inputs:
    data (numpy array): data to be fitted, assumed in the shape (npix,npix,nfreq)
    frequency (numpy array): array of the frequency of each slice of the data cube with length nfreq
    
    Outputs
    model (numpy array): model produced by GPR algorithm in the shape (npix,npix,nfreq)
    residuals (numpy array): residuals produced by data - model in the shape (npix,npix,nfreq)
    """
    #choose kernel
    # kernel for the smooth foreground:
    kern_sfg = GPy.kern.RBF(1)
    #mixing kernel
    kern_mix = GPy.kern.Matern32(1)
    #ex kernel
    kern_ex = GPy.kern.Matern52(1)
    # kernel for the HI cosmological signal:
    kern_21 = GPy.kern.Exponential(1)
    
    #set lengthscales to ensure the kernels fit to the correct part of the signal based on the data in Mertens et al (2020)
    kern_sfg.lengthscale.constrain_bounded(10,100)
    kern_21.lengthscale.constrain_bounded(0.1,1.2)
    kern_mix.lengthscale.constrain_bounded(1,10)
    kern_ex.lengthscale.constrain_bounded(0.2,8)
    kern_fg = kern_sfg + kern_ex + kern_mix

    gpr_result = fg.GPRclean(data, frequency, kern_fg, kern_21, NprePCA=0, num_restarts=10,
                                              noise_data=None, heteroscedastic=False, zero_noise=True, invert=False)

    model_gpr = gpr_result.fgfit 
    resids_gpr = data - model_gpr

    return model_gpr, resids_gpr
