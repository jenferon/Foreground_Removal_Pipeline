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
def mu_powerspec(data,kbins,mubins,frequency):
  box_dims = find_box_dims(frequency, FoV)
  
  
def powerspec_1d(data,kbins,frequency,FoV):

  box_dims = find_box_dims(frequency, FoV)

  p_lc_eor_res, k_lc_eor_res, n_lc_res = t2c.power_spectrum_1d(resids_LC_FG[:,:,i0:i1], kbins=kbins, box_dims=box_dims, binning =  'log', return_n_modes=True)
  d_lc = (p_lc_eor*k_lc_eor**3)/(2*np.pi**2)
  err_rsd_res= np.empty_like(n_rsd_res)

for ii in range(0,n_rsd_res.shape[0]):
    err_lc_res[ii] = d_lc_res[ii]/np.sqrt(n_lc_res[ii])

def cyclindircal_powerspec(data,kbins,frequency):

   box_dims = find_box_dims(frequency, FoV)
p_lc_eor_res, k_lc_eor_res, n_lc_res = t2c.power_spectrum_1d(resids_LC_FG[:,:,i0:i1], kbins=kbins, box_dims=box_dims, binning =  'log', return_n_modes=True)
