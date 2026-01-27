import bdsf

base='/home/ppxjf3/repos/Update_FG_Pipeline_temp/'

def run_pbdsf(img=base+'initial_uniform-image.fits'):
    """
    A function to run the source finding software PyBDSF and output a catalogue of the brightest point sources
    
    img (str): file location of image to process 
    """
    img = bdsf.process_image(img,  rms_box=(30,10), thresh_isl=20, thresh_pix=15)
    img.export_image(outfile=base+'bright_point_sources.fits', clobber=True, img_type='guass')
    img.write_catalog(format='csv', clobber=True, outfile='bright_points_cat.txt')
    
    
if __name__ == "__main__":
    run_pbdsf()
