#!/usr/bin/python
# Example settings script for OSKAR. Originally written by Fred Dulwich and Benjamin Mort with later additions made by Emma Chapman. This can be used as part of a submission script or as a guide for using the GUI.

import matplotlib.pyplot as plt
import oskar

base = '/gpfs01/home/ppxjf3/peculiar_vel/data/NOISE/'

def create_dir(name):
    import os
    if not os.path.isdir(name):
        os.makedirs(name)

def setup_data_dirs(dir_list):
    for dir_ in dir_list:
        create_dir(base + dir_)

def fov_to_cellsize(fov_deg, size):
    """Convert image FoV and size along one dimension in pixels to cellsize in arcseconds.
    Arguments:
    fov_deg -- Image FoV, in degrees
    size    -- Image size in one dimension in pixels
    Return:
    Image cellsize, in arcseconds
    """
    import numpy as np
    rmax = np.sin(fov_deg / 2.0 * (np.pi / 180.0))
    inc  = rmax / (0.5 * size)
    return np.arcsin(inc) * ((180.0 * 3600.0) / np.pi), inc

def image_lm_grid(size, fov_rad):
    import numpy as np
    _,inc = fov_to_cellsize(fov_rad*(180.0/np.pi), size)
    lm = np.arange(-size/2, size/2)*inc
    [l,m]=np.meshgrid(-lm,lm)
    return l,m

def lm_to_apparent_ra_dec(l, m, ra0_rad, dec0_rad):
    import numpy as np
    n = np.sqrt(1.0 - l*l - m*m)
    sinDec0 = np.sin(dec0_rad)
    cosDec0 = np.cos(dec0_rad)
    dec_rad = np.arcsin(n * sinDec0 + m * cosDec0)
    ra_rad = ra0_rad + np.arctan2(l, n * cosDec0 - m * sinDec0)
    return ra_rad, dec_rad

def image_coords(size, fov_rad, ra0_rad, dec0_rad):
    l, m = image_lm_grid(size, fov_rad)
    return lm_to_apparent_ra_dec(l, m, ra0_rad, dec0_rad)

def slice_to_osm(sky_root_name, fov_deg, freq, ra0_deg, dec0_deg, \
        osm_filename, upscale_factor, max_bl,save_fits=False):
    # Convert a fits simulation image into an OSKAR source model (OSM)    
    import numpy as np
    from astropy.io import fits
    import os
    from PIL import Image
    ra0_rad = ra0_deg * np.pi / 180.0
    dec0_rad = dec0_deg * np.pi / 180.0
    slice_filename = base+'models/%s.fits' % (sky_root_name)

    # Open the FITS image data slice
    # -------------------------------------------------------------------------
    hdulist = fits.open(slice_filename)
    slice = hdulist[0].data
    assert(freq >= 0)
    hdulist.close()

    # -------------------------------------------------------------------------
    im0 = np.array(slice, dtype=np.float32)

    # Rescale the raw image (im0) to obtain im1
    # -------------------------------------------------------------------------
    old_size = im0.shape[0]
    freqHz = freq*1.0e6
    size = int(np.ceil(upscale_factor * old_size))
    im_ = Image.frombuffer('F', im0.shape, im0.tostring(), 'raw', 'F', 0, 1)
    im_ = im_.resize((size,size), Image.NEAREST)
    im1 = np.array(im_.getdata(), dtype=np.float32).reshape((size, size))

    # Convert the rescaled image to an OSM file.
    # -------------------------------------------------------------------------
    # Obtain list of image coordinates for the rescaled map pixels
    ra, dec = image_coords(size, fov_deg * np.pi/180.0, ra0_rad, dec0_rad)
    ra  *= 180.0/np.pi
    dec *= 180.0/np.pi

    num_pixels = int(size*size)
    sky = np.zeros((num_pixels, 3))
    sky[:, 0] = ra.reshape((num_pixels,))
    sky[:, 1] = dec.reshape((num_pixels,))
    sky[:, 2] = im1.reshape((num_pixels,))

    # Remove sources with amplitude == 0.0
    sky = sky[sky[:, 2] != 0.0, :]

    # Convert to Jy/pixel
    old_cellsize,_ = fov_to_cellsize(fov_deg, old_size)
    new_cellsize,_ = fov_to_cellsize(fov_deg, size)
    # new pixel size is smaller than the old pixel size so the pixel area ratio
    # will be < 1 for all cases of upscaling.
    kB = 1.3806488e-23
    c0 = 299792458.0
    pixel_area = np.power(((new_cellsize / 3600.0) * np.pi / 180.0), 2)
    # Convert from brightness temperature in K to Jy/Pixel
    # http://www.iram.fr/IRAMFR/IS/IS2002/html_1/node187.html
    sky[:, 2] *= 2.0 * kB * 1.0e26 * pixel_area * np.power(freqHz/c0,2)

    if (os.path.exists(osm_filename)):
        os.remove(osm_filename)

    if (np.__version__ == '1.4.1'):
        np.savetxt(osm_filename, sky, fmt='%.10e, %.10e, %.10e')
    else:
        np.savetxt(osm_filename, sky, fmt='%.10e, %.10e, %.10e', \
            header = \
            'Frequency = %e MHz\n'\
            'Number of sources = %i\n'\
            'Cellsize = %f arcsec (pixel separation at centre)\n'\
            'RA0 = %f\n' \
            'Dec0 = %f\n' \
            % (freqHz, len(sky), new_cellsize, ra0_deg, dec0_deg))

    # Save FITS maps of the selected channel slice
    if save_fits == True:
        outpath = os.path.dirname(slice_filename)

        img = '%s/IMG_%s_K_%05.1f.fits' % (outpath, sky_root_name, freq)
        if os.path.exists(img): os.remove(img)
        im0 = np.reshape(im0, (1, im0.shape[0], im0.shape[1]))
        #im0[0,256,256] = 100
        hdu = fits.PrimaryHDU(im0)
        hdulist = fits.HDUList([hdu])
        hdr = hdulist[0].header
        hdr['BUNIT']  = ('K', 'Brightness unit')
        hdr['CTYPE1'] = 'RA---SIN'
        hdr['CRVAL1'] = ra0_deg
        hdr['CDELT1'] = -old_cellsize / 3600.0
        hdr['CRPIX1'] = im0.shape[1] / 2 + 1 # WARNING! Assumes even image dims
        hdr['CTYPE2'] = 'DEC--SIN'
        hdr['CRVAL2'] = dec0_deg
        hdr['CDELT2'] = old_cellsize / 3600.0
        hdr['CRPIX2'] = im0.shape[1] / 2 + 1 # WARNING! Assumes even image dims
        hdr['CTYPE3'] = 'FREQ'
        hdr['CRVAL3'] = freqHz
        hdr['CDELT3'] = 1
        hdr['CRPIX3'] = 1
        hdulist.writeto(img)
        hdulist.close()

        rescaled_img = '%s/IMG_%s_K_rescaled_%05.1f.fits' % (outpath, sky_root_name, freq)
        if os.path.exists(rescaled_img): os.remove(rescaled_img)
        im1 = np.reshape(im1, (1, im1.shape[0], im1.shape[1]))
        hdu = fits.PrimaryHDU(im1)
        hdulist = fits.HDUList([hdu])
        hdr = hdulist[0].header
        hdr['BUNIT']  = ('K', 'Brightness unit')
        hdr['CTYPE1'] = 'RA---SIN'
        hdr['CRVAL1'] = ra0_deg
        hdr['CDELT1'] = -new_cellsize / 3600.0
        hdr['CRPIX1'] = im1.shape[1] / 2 + 1 # WARNING! Assumes even image dims
        hdr['CTYPE2'] = 'DEC--SIN'
        hdr['CRVAL2'] = dec0_deg
        hdr['CDELT2'] = new_cellsize / 3600.0
        hdr['CRPIX2'] = im1.shape[1] / 2 + 1 # WARNING! Assumes even image dims
        hdr['CTYPE3'] = 'FREQ'
        hdr['CRVAL3'] = freqHz
        hdr['CDELT3'] = 1
        hdr['CRPIX3'] = 1
        hdulist.writeto(rescaled_img)
        hdulist.close()

        # Convert to Jy/Beam
        FWHM = 1.22 * (c0/freqHz) / max_bl # Radians
        beam_area = (np.pi * FWHM**2) / (4.0*np.log(2))
        im2 = im1 * 2.0 * kB * 1.0e26 * beam_area * (freqHz/c0)**2
        clean_component_img = '%s/IMG_%s_Jy_per_beam_%05.1f.fits' % (outpath,sky_root_name, freq)
        if os.path.exists(clean_component_img): os.remove(clean_component_img)
        hdu = fits.PrimaryHDU(im2)
        hdulist = fits.HDUList([hdu])
        hdr = hdulist[0].header
        hdr['BUNIT']  = ('Jy/beam', 'Brightness unit')
        hdr['CTYPE1'] = 'RA---SIN'
        hdr['CRVAL1'] = ra0_deg
        hdr['CDELT1'] = -new_cellsize / 3600.0
        hdr['CRPIX1'] = im1.shape[1] / 2 + 1 # WARNING! Assumes even image dims
        hdr['CTYPE2'] = 'DEC--SIN'
        hdr['CRVAL2'] = dec0_deg
        hdr['CDELT2'] = new_cellsize / 3600.0
        hdr['CRPIX2'] = im1.shape[1] / 2 + 1 # WARNING! Assumes even image dims
        hdr['CTYPE3'] = 'FREQ'
        hdr['CRVAL3'] = freqHz
        hdr['CDELT3'] = 1
        hdr['CRPIX3'] = 1
        hdulist.writeto(clean_component_img)
        hdulist.close()

        # Convert to Jy/Pixel
        im3 = im1 * 2.0 * kB * 1.0e26 * pixel_area * (freqHz/c0)**2
        sky_model_img = '{0}/IMG_{1}_Jy_per_pixel_{2}.fits'.format(outpath,sky_root_name, freq)
        if os.path.exists(sky_model_img): os.remove(sky_model_img)
        hdu = fits.PrimaryHDU(im3)
        hdulist = fits.HDUList([hdu])
        hdr = hdulist[0].header
        hdr['BUNIT']  = ('Jy/pixel', 'Brightness unit')
        hdr['CTYPE1'] = 'RA---SIN'
        hdr['CRVAL1'] = ra0_deg
        hdr['CDELT1'] = -new_cellsize / 3600.0
        hdr['CRPIX1'] = im1.shape[1] / 2 + 1 # WARNING! Assumes even image dims
        hdr['CTYPE2'] = 'DEC--SIN'
        hdr['CRVAL2'] = dec0_deg
        hdr['CDELT2'] = new_cellsize / 3600.0
        hdr['CRPIX2'] = im1.shape[1] / 2 + 1 # WARNING! Assumes even image dims
        hdr['CTYPE3'] = 'FREQ'
        hdr['CRVAL3'] = freqHz
        hdr['CDELT3'] = 1
        hdr['CRPIX3'] = 1
        hdulist.writeto(sky_model_img)
        hdulist.close()

    print ('  Input file           = {}' .format(slice_filename))
    print( '  Frequency            = {} MHz' .format (freqHz / 1.0e6))
    print ('  Slice image size      = {}'.format( im0.shape[1]))
    print ('  No. sources          = {0} [{1} {2}]' .format(len(sky), (size*size)-len(sky), 'removed (==0.0)'))
    print ('  Output sky model     ={}'.format( osm_filename))
    print ('  Writing FITS images  ={}'.format( save_fits))
    if save_fits == True:
        print ('  Beam area            ={}'.format( beam_area))
        print ('  Pixel area           ={}'.format(pixel_area))

    return len(sky)

def set_setting(ini, key, value):
    from subprocess import call
    open(ini, 'a').close()
    call(["oskar_sim_interferometer", "--set", ini, key, str(value)])

def run_interferometer(ini, verbose=True):
    from subprocess import call
    if verbose:
        call(["oskar_sim_interferometer", ini])
    else:
        call(["oskar_sim_interferometer", "-q", ini])

def dict_to_settings(settings_dict, filename):
    for group in sorted(settings_dict.keys()):
        for key in sorted(settings_dict[group].keys()):
            key_ = group+key
            value_ = settings_dict[group][key]
            set_setting(filename, key_, value_)

def require_oskar_version(version):
    import subprocess
    import re
    try:
        subprocess.call(['oskar_sim_interferometer', '--version'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE);
    except OSError:
        raise Exception('OSKAR not found. Check your PATH settings.')
    proc = subprocess.Popen('oskar_sim_interferometer --version', \
        stdout=subprocess.PIPE, shell=True)
    (out,err) = proc.communicate()
    out = out.strip('\n\r')
    ver = re.split('\.|-', out)
    ver[0] = int(ver[0])
    ver[1] = int(ver[1])
    ver[2] = int(ver[2])
    sVersion = '{0}.{1}.{2}' .format(version[0], version[1], version[2])
    if len(version) == 4: sVersion='{0}-{1}' .format(sVersion, version[3])
    sVer = '{0}.{1}.{2}'.format(ver[0], ver[1], ver[2])
    if len(ver) == 4: sVer='{0}-{1}'.format(sVer, ver[3])
    failMsg = "ERROR: This script requires OSKAR {0} [found version {1}]." .format(sVersion, out)
    if (len(ver)!=len(version)):
        print( failMsg)
    for i in range(0, len(ver)):
        if (ver[i] != version[i]): print( failMsg)
    return ver

# bw = Bandwidth in Hz, and obs_length in seconds.
# based on http://www.skatelescope.org/uploaded/59513_113_Memo_Nijboer.pdf
def evaluate_noise_rms_Jy(freqHz, bw, obs_length):
    import numpy as np
    c0 = 299792458.
    kB = 1.3806488e-23
    lambda_ = c0 / freqHz

    # Values from Peter Dewdney's spreadsheet.
    A_sparse = lambda_**2 / 3
    A_physical = 1.5625          # Element physical size within HBA tile (5mx5m tile / 16 antenna)
    eta_s = 0.9                  # System efficiency
    station_size = 24*16         # Number of antennas per station (core station, single patch = 24 tiles of 16 antenna)
    T_recv = 140                 # http://arxiv.org/pdf/0901.3359v1.pdf (140-180K)
    #T_recv = 180                # http://arxiv.org/pdf/1104.1577.pdf (140-180K)

    # Get effective area per dipole antenna, A_eff
    A_eff = np.minimum(A_sparse, A_physical)

    # Get system temperature.
    T_sky = 60.0 * np.power(lambda_, 2.55) # "Standard" empirical T_sky
    T_sys = T_sky + T_recv

    # Get RMS noise per baseline for single polarisation.
    SEFD_station = ((2.0 * kB * T_sys) / (eta_s * A_eff * station_size))
    SEFD_station *= 1e26 # Convert to Jy
    sigma_pq = SEFD_station / np.sqrt(2.0 * bw * obs_length)

    print('Noise:')
    print( '- Frequency          = {0} MHz (wavelength = {1} m)'.format(freqHz/1.e6, lambda_))
    print('- Tsys               = {} K'.format(T_sys))
    print('- SEFD (station)     = {} kJy'.format(SEFD_station*1e-3))
    print('- Observation length = {} hours'.format(obs_length / 3600.0))
    print('- sigma_pq           = {} mJy'.format(sigma_pq*1e3))

    return sigma_pq

def create_settings(freqHz, sky_model_name, ra0_deg, dec0_deg, ms_name,
    start_time, obs_length, num_time_steps, uvmax, add_noise, noise_rms_Jy, noise_seed, telescope, telescope_model):
    s = {}
    s['simulator/'] = {
        'max_sources_per_chunk':16384,   #65536 if just I 
        'double_precision':False,
        'keep_log_file':'true',
        'use_gpus': 'true'
    }
   
    if (telescope == 'SKA'):
        s['telescope/'] = {
            'input_directory':str(telescope_model)+'.tm',
            'pol_mode':'Full',
            'normalise_beams_at_phase_centre':'true',
            'allow_station_beam_duplication':'true',
            'aperture_array/array_pattern/enable':'true',
            'aperture_array/element_pattern/functional_type':'Dipole',
            'aperture_array/element_pattern/dipole_length':0.5,
            'aperture_array/element_pattern/dipole_length_units':'Wavelengths',
            # ------
            #'station_type':'Isotropic',
            # -----
            'station_type':'Aperture',
            #'station_type':'Gaussian beam',
            #'gaussian_beam/fwhm_deg':15.1	,
            #'gaussian_beam/ref_freq_hz':145e6,
            #'station_type':'Gaussian beam',
        }

    s['observation/'] = {
        'phase_centre_ra_deg':ra0_deg,
        'phase_centre_dec_deg':dec0_deg,
        'start_frequency_hz':freqHz,
        'num_channels':1,
        'start_time_utc':start_time,
        'length':obs_length,
        'num_time_steps':num_time_steps
    }
    if add_noise == False:
        s['sky/'] = {
            'oskar_sky_model/file':sky_model_name,
            'advanced/apply_horizon_clip':'false'
        }    
        s['interferometer/'] = {
	    'oskar_vis_filename':ms_name[:-3]+'.vis',
            'channel_bandwidth_hz':183e3,
            'time_average_sec':60.0,
            #'ms_filename':ms_name,
            'uv_filter_min':'min',
            'uv_filter_max':'max',
            'uv_filter_units':'Wavelengths',
        }
    
    if add_noise == True:
        noise = {
            'oskar_vis_filename':ms_name[:-3]+'.vis', #only if you want to output .vis as well as .MS
            'noise/enable':add_noise,
            'noise/seed':noise_seed,
#            'noise/values':'RMS flux density',
            'noise/rms':'Range',
            'noise/rms/start':noise_rms_Jy,
            'noise/rms/end':noise_rms_Jy,
            'noise/freq':'Range',
            'noise/freq/number':1,
            'noise/freq/start':freqHz,
            'noise/freq/inc':0
        }
        s['interferometer/'] = {
             'oskar_vis_filename':ms_name[:-3]+'.vis',            
             'channel_bandwidth_hz':183e3,
             'time_average_sec':10.0,
             'ms_filename':ms_name,
             'uv_filter_min':'min',
             'uv_filter_max':'max',
             'uv_filter_units':'Wavelengths',
         }
        s['interferometer/'].update(noise)
    return s

if __name__ == '__main__':
    import sys
    import os
    import numpy as np

    freq = float(sys.argv[1]) #MHz
    mode = int(sys.argv[2])
    sky_root_name = sys.argv[3]
    fov_deg  = float(sys.argv[4])#5.0
    num_pixels_side = int(sys.argv[5])#512
    field = sys.argv[7]
    telescope = sys.argv[6]
    telescope_model = sys.argv[8]
    max_bl = float(sys.argv[9])
    generate_data = False
    run_sim = True 

    if (field == 'EOR0'):
#       MWA EOR0 field
        ra0_deg  = 0.0
        dec0_deg = -27.0

    upscale_factor = 2.0
    noise_obs_length = 600.0 * 3600.0
    noise_bw = 3.66e6
    if ((mode == 1) or (mode == 2)):
        add_noise = False
    elif mode == 3:
        add_noise = True
    else:
        print( 'ERROR: Invalid mode option.')
        os.exit(1)

    if generate_data== True and mode == 1: #makes osm file
        # Generate sky model from slice of FITS slice.
        osm_filename  = base+'osm/{}.osm'.format(sky_root_name)
        print( 'here is the filename {}'.format(osm_filename))
        slice_to_osm(sky_root_name, fov_deg, freq, ra0_deg, dec0_deg, \
            osm_filename, upscale_factor, max_bl, True)
                
    if run_sim == True:
        setup_data_dirs(['vis','ini'])   	      
        # Set up parameters.
        freqHz = freq*1.0e6
        ms_name = base+'vis/{0}_{1}_{2}_{3}_{4}_{5}.ms'.format(sky_root_name, telescope, telescope_model, field, fov_deg, num_pixels_side)
        ini = base+'ini/{0}_{1}_{2}_{3}_{4}_{5}.ini' .format(sky_root_name, telescope, telescope_model, field, fov_deg, num_pixels_side)
        osm_filename = base+'osm/{}.osm' .format(sky_root_name)
        if (telescope == 'SKA'):
            # this start time keeps EOR0 above horizon for 12 hours. 
            start_time = '2000-01-01 03:30:00'
        obs_length = 12.0 * 3600.0
        if add_noise == False:
            obs_interval = 60.0
        if add_noise == True:
            obs_interval = 10.0
        num_time_steps = int(np.ceil(obs_length / obs_interval))
        uvmax = (upscale_factor * num_pixels_side) / \
            (2.0 * fov_deg * np.pi/180.0)
        noise_rms_Jy = evaluate_noise_rms_Jy(freqHz, noise_bw, noise_obs_length)

        # Create settings file.
        noise_seed = int(sys.argv[10])
        print (noise_seed)
        s = create_settings(freqHz, osm_filename, ra0_deg, dec0_deg,
            ms_name, start_time, obs_length, num_time_steps, uvmax,
            add_noise, noise_rms_Jy, noise_seed, telescope, telescope_model)
        dict_to_settings(s, ini)
        print ('Running simulation for freq = {:.4f} MHz'.format(freqHz/1.0e6))
        run_interferometer(ini)

        precision = 'single'
        fov_deg=1.0
        image_size=100
        weighting="natural"

        image_type="I"
        imager = oskar.Imager()
        ms=".vis"
        ms_name = base + 'vis/%s_%s_%s_%s_%03.1f_%03d.vis' % (sky_root_name, telescope, telescope_model, field, fov_deg, num_pixels_side)
        out_name = base + '%s_%s_%s_%s_%03.1f_%04d_%s_%s'% (sky_root_name, telescope, telescope_model, field, fov_deg, num_pixels_side,weighting,image_type)
        imager.set(image_type=image_type, fov_deg=fov_deg, image_size=num_pixels_side)
        imager.set(input_file=ms_name, output_root=out_name)
        #imager.set(uv_filter_min=1000) #set max baseline used - hopefully in units of m 
        output = imager.run(return_images=1)
        image = output["images"][0]

        # Render the image using matplotlib and save it as a PNG file.
        #im = plt.imshow(image, cmap="jet")
        #plt.gca().invert_yaxis()
        #plt.colorbar(im)
        #plt.savefig(out_name+'.png')

        #image_type="PSF"
        #imager = oskar.Imager()
        #ms=".vis"
        #ms_name = base + 'vis/%s_%s_%s_%s_%03.1f_%03d.vis' % (sky_root_name, telescope, telescope_model, field, fov_deg, num_pixels_side)
        #out_name = base + '%s_%s_%s_%s_%03.1f_%04d_%s_%s'% (sky_root_name, telescope, telescope_model, field, fov_deg, num_pixels_side,weighting,image_type)
        #imager.set(image_type=image_type, fov_deg=fov_deg, image_size=num_pixels_side)
        #imager.set(input_file=ms_name, output_root=out_name)
        #output = imager.run(return_images=1)
        #image = output["images"][0]

        # Render the image using matplotlib and save it as a PNG file.
        #im = plt.imshow(image, cmap="jet")
        #plt.gca().invert_yaxis()
        #plt.colorbar(im)
        #plt.savefig(out_name+'.png')

