import py21cmsense as p21c
from py21cmsense import Observation, Observatory, PowerSpectrum, beam

def noise():
  
  #making hera limits
  hera_ants = p21c.antpos.hera(hex_num=11, row_separation=12.12 * u.m)
  assert hera_ants.shape == (331, 3)
  
  nu = 145
  c = 2.998e8*u.m/u.s
  beam = p21c.GaussianBeam(
      frequency=nu * u.MHz, dish_size=7 * (c / (150 * u.MHz)).to("m")
  )
  hera = p21c.Observatory(
      antpos=hera_ants,
      beam=beam,
      latitude=0.6707845 * u.rad,
      Trcv=100 * u.K,
      beam_crossing_time_incl_latitude=False,
  )
  
  
  #observation
  observation_params = {}
  observation_params["ndays"] = 166.7
  observation_params["cosmo"] = cosmo
  observation_params["h"] = cosmo.H0.value / 100.0
  observation_params["freq_bands"] = nu
  observation_params["redshifts"] = zs
  observation_params["time_per_day_hrs"] = 6.0
  observation_params["bandwidth"] = 10e6
  
  from astropy.cosmology import Planck15
  
  obs = p21c.Observation(
      observatory=hera,
      tsky_amplitude=60 * u.K,
      tsky_ref_freq=300 * u.MHz,
      spectral_index=2.6,
      n_days=180,
      time_per_day=6 * u.hour,
      bandwidth=8 * u.MHz,
      n_channels=82,
      integration_time=60 * u.s,
      lst_bin_size=beam.at(150 * u.MHz).fwhm.value * 12 / np.pi * 3600 * u.s,
      use_approximate_cosmo=True,
      cosmo=Planck15.clone(H0=70.0, Om0=0.266),
  )
  xx=np.logspace(-1., 2.5, 30)
  print(xx)
  kperp_edges= u.Quantity(xx*0.71, "1/Mpc")
  sense_moderate =PowerSpectrum(foreground_model="foreground_free", horizon_buffer=0.0/u.Mpc,
              observation=obs,
          ).at_frequency(nu * u.MHz)
  
  #sense1d_both = sense_moderate.calculate_sensitivity_1d_binned(thermal=True, sample=False, k=kperp_edges)
  sense1d = sense_moderate.calculate_sensitivity_1d(thermal=True, sample=True)
  xx = xx*0.71

  return sense_moderate.k1d.value*0.7, sense1d
