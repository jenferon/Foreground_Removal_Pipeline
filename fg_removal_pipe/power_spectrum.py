import tools21cm as t2c

kbins=15
mubins=15

  
i0 = 0
i1 = 101  
def find_box_dims(frequency, FoV):
  L_para, L_perp = get_lengths(frequency[i0], frequency[i1], (frequency[i0]+frequency[i1])/2, FoV)
  box_dims = [L_perp, L_perp, L_para]

  return box_dims

L_para, L_perp = get_lengths(freq[i0], freq[i1], (freq[i0]+freq[i1])/2, 1.0)
box_dims = [L_perp, L_perp, L_para]
def mu_powerspec(data,nkbins,nmubins,frequency):
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

  box_dims = find_box_dims(frequency, FoV)

  p_lc_eor_res, k_lc_eor_res, n_lc_res = t2c.power_spectrum_1d(resids_LC_FG[:,:,i0:i1], kbins=kbins, box_dims=box_dims, binning =  'log', return_n_modes=True)
  d_lc = (p_lc_eor*k_lc_eor**3)/(2*np.pi**2)
  err_rsd_res= np.empty_like(n_rsd_res)

for ii in range(0,n_rsd_res.shape[0]):
    err_lc_res[ii] = d_lc_res[ii]/np.sqrt(n_lc_res[ii])

def cyclindircal_powerspec(data,kbins,frequency):

   box_dims = find_box_dims(frequency, FoV)
  
  pp, kper, kpar= t2c.power_spectrum_2d(data, kbins=bins, box_dims=box_dims, return_modes=False)

  #normalise to dimensionless power spectrum
  for ii in range(0,len(kper)):
      for jj in range(0,len(kpar_RSD)):
          pp[ii,jj] = (pp[ii,jj]*np.sqrt(kper[ii]**2+kpar[jj]**2)**3)/(2*np.pi**2)
  
  fp = interpolate.interp2d(kper, kpar, pp.T, kind='linear')
  CC = fp(kper,kpar)
  norm = colors.LogNorm(vmin=CC[np.isfinite(CC)].min(), vmax=CC[np.isfinite(CC)].max()) if plotting_scale['z']=='log' else None 

  return kper, kpar, CC, norm
