import tools21cm as t2c
import numpy as np
from scipy import interpolate
import matplotlib.colors as colors
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

cosmo = FlatLambdaCDM(H0=71 * u.km / u.s / u.Mpc, Om0=0.27)
plotting_scale={'x': 'log', 'y': 'log', 'z': 'log'}


def zFromNu(nu):
  """
  Convert frequency of 21cm line to redshift
    
  Input: nu [MHz]
  """
  nu21 = 1.420405e3  #MHz
  return nu21/nu - 1.0
  

def get_lengths(nu0, nu1, theta):
  """
  Docstring for get_lengths
    
  :param nu0: Description
  :param nu1: Description
  :param FoV: Description
  """
  nu_mid = (nu0+nu1)/2
  z_lo = zFromNu(nu0)
  z_mid = zFromNu(nu_mid)
  z_hi = zFromNu(nu1)
    
  L_para = cosmo.comoving_distance(z_lo) - cosmo.comoving_distance(z_hi)
  L_perp = cosmo.comoving_distance(z_mid) * (np.pi * theta / 180.0)
  return L_para/u.Mpc, L_perp/u.Mpc


def find_box_dims(frequency, FoV):
  L_para, L_perp = get_lengths(frequency[0], frequency[-1], FoV)
  box_dims = [L_perp, L_perp, L_para]

  return box_dims

def mu_powerspec(data, nkbins, nmubins, frequency, FoV):
  """
  Calculating the power spectrum in mu and k

  Inputs:
  data (nunmpy array):
  nkbins (int)
  nmubins (int):
  frequency (numpy array):

  Outputs:
  dk (numpy array)
  kbins (numpy array)
  mubins (numpy array)
  err (numpy array)

  """
  box_dims = find_box_dims(frequency, FoV)
  PP, mubins, kbins, nmode = t2c.power_spectrum_mu(data, los_axis = 2, box_dims=box_dims, mubins=nmubins,kbins=nkbins, exclude_zero_modes=True,return_n_modes=True,absolute_mus=False)
  dk = (PP*kbins**3)/(2*np.pi**2)

  #calculate the 1/root(n) error
  err= np.empty_like(nmode)

  for ii in range(0,nmode.shape[0]):
      for jj in range(0,nmode.shape[1]):
          err[ii, jj] = dk[ii, jj]/np.sqrt(nmode[ii, jj])

  return dk, kbins, mubins, err

def powerspec_1d(data,kbins,frequency,FoV):
  """
  Docstring for powerspec_1d
  
  :param data: Description
  :param kbins: Description
  :param frequency: Description
  :param FoV: Description
  """

  box_dims = find_box_dims(frequency, FoV)

  p, k, n = t2c.power_spectrum_1d(data, kbins=kbins, box_dims=box_dims, binning =  'log', return_n_modes=True)
  d = (p*k**3)/(2*np.pi**2)
  err= np.empty_like(n)

  for ii in range(0,n.shape[0]):
    err[ii] = d[ii]/np.sqrt(n[ii])
  
  return d, k, err
  

def cyclindircal_powerspec(data, kbins, frequency, FoV):
  """
  Docstring for cyclindircal_powerspec
  
  :param data: Description
  :param kbins: Description
  :param frequency: Description
  :param FoV: Description
  """

  box_dims = find_box_dims(frequency, FoV)
  
  pp, kper, kpar= t2c.power_spectrum_2d(data, kbins=kbins, box_dims=box_dims, return_modes=False)

  #normalise to dimensionless power spectrum
  for ii in range(0,len(kper)):
      for jj in range(0,len(kpar)):
          pp[ii,jj] = (pp[ii,jj]*np.sqrt(kper[ii]**2+kpar[jj]**2)**3)/(2*np.pi**2)
  
  fp = interpolate.interp2d(kper, kpar, pp.T, kind='linear')
  CC = fp(kper,kpar)
  norm = colors.LogNorm(vmin=CC[np.isfinite(CC)].min(), vmax=CC[np.isfinite(CC)].max()) if plotting_scale['z']=='log' else None 

  return kper, kpar, CC, norm
