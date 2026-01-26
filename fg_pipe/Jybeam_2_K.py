import numpy as np

def TFromJyBeam(jb, theta_max, theta_min, nu):
    """
    Convert from Jy/Beam to brightness temperature
    
    https://science.nrao.edu/facilities/vla/proposing/TBconv
    
    Units: theta_min, theta_max [deg]
           jb [Jy/beam]
           nu [MHz]
           tbright [K]
           
    (definitions use arcsec, mJy/beam and GHz, so convert those in code)
    """
    
    tbright = 1.222e3 * (jb * 1e3)
    tbright /= np.power(nu / 1.0e3, 2.0) * (theta_max * theta_min * 3600**2)
    
    return tbright