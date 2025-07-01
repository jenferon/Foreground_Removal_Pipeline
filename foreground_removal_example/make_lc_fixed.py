import py21cmfast as p21c
from py21cmfast import cache_tools
import numpy as np
from astropy.io import fits
from scipy import interpolate
import os

class lightcone(object): 
	def __init__ (self, BOX_LEN, HII_DIM, z_start, n_z, z_step, angle, freq_start, nfreq, freq_step):

                if not isinstance(BOX_LEN, float) and not isinstance(BOX_LEN, int):
                        print(type(BOX_LEN))
                        raise TypeError("variable BOX_LEN is not a int/float")

                else:
                     	self.BOX_LEN = BOX_LEN

                if not isinstance(HII_DIM, float) and not isinstance(HII_DIM, int):
                        print(type(HII_DIM))
                        raise TypeError("variable HII_DIM is not a int/float")

                else:
                     	self.HII_DIM = HII_DIM
                
                if not isinstance(z_start, float) and not isinstance(z_start, int):
                        print(type(z_start))
                        raise TypeError("variable z_start is not a int/float")

                else:
                     	self.z_start = z_start     	
               
                if not isinstance(n_z, float) and not isinstance(n_z, int):
                        print(type(n_z))
                        raise TypeError("variable n_z is not a int/float")

                else:
                     	self.n_z = n_z
                     	
                if not isinstance(z_step, float) and not isinstance(z_step, int):
                        print(type(z_step))
                        raise TypeError("variable z_step is not a int/float")

                else:
                     	self.z_step = z_step
                     	
                if not isinstance(angle, float) and not isinstance(angle, int):
                        print(type(angle))
                        raise TypeError("variable angle is not a int/float")

                else:
                     	self.angle = angle
                  
                if not isinstance(freq_start, float) and not isinstance(freq_start, int):
                        print(type(freq_start))
                        raise TypeError("variable freq_start is not a int/float")

                else:
                     	self.freq_start = freq_start     	
               
                if not isinstance(nfreq, float) and not isinstance(nfreq, int):
                        print(type(nfreq))
                        raise TypeError("variable nfreq is not a int/float")

                else:
                     	self.nfreq = nfreq
                     	
                if not isinstance(freq_step, float) and not isinstance(freq_step, int):
                        print(type(freq_step))
                        raise TypeError("variable freq_step is not a int/float")

                else:
                     	self.freq_step = freq_step
                     	
                if not os.path.exists('21cmFAST-cache'):
                        os.mkdir('21cmFAST-cache')

                p21c.config['direc'] = '21cmFAST-cache'
                cache_tools.clear_cache(direc="21cmFAST-cache")
	def initial_conditions(self, SIGMA_8=0.82, OMb=0.046, OMm=0.28,POWER_INDEX=0.96, hlittle=0.73):
		initial_conditions = p21c.initial_conditions(user_params = {"N_THREADS":40, "HII_DIM":self.HII_DIM, "BOX_LEN":self.BOX_LEN},
		cosmo_params = p21c.CosmoParams(SIGMA_8=0.82, OMb=0.046, OMm=0.28,POWER_INDEX=0.96, hlittle=0.73), random_seed=54321)
		
		self.init_condits = initial_conditions
		return initial_conditions
	
	def make_cubes(self):
		#make coeval cubes from 6 to 28
		self.redshift = self.z_start + self.z_step * np.arange(self.n_z)
		cosmo_params = self.init_condits.cosmo_params

		brightness_temp = []
		for i in range(0,len(self.redshift)):
        		coeval = p21c.run_coeval(redshift=self.redshift[i],flag_options= {"USE_TS_FLUC": True}, init_box=self.init_condits)
        		brightness_temp.append(coeval.brightness_temp)
        		print('run coeval' + str(self.redshift[i]))
		self.Tb=brightness_temp
                #return brightness_temp

	def make_lc(self):
        	#field of view angle
		angle_r = self.angle*(np.pi/180)

		#make lightcone
		#put slices of coeval cubes together to make lightcone
		self.freq =  self.freq_start + self.freq_step * np.arange(self.nfreq)

		#define params
		write_to_file =np.array(['freq', 'z', 'z_box', 'k'])
		box_len =  self.BOX_LEN #define length of coeval box in MPc
		cell_num = self.HII_DIM #define cell number of coeva box
		delta_x = box_len/cell_num
		cosmo_params = self.init_condits.cosmo_params
		x_min =cosmo_params.cosmo.comoving_distance(6).value
		x_max = cosmo_params.cosmo.comoving_distance(28).value

		lc = np.empty((cell_num,cell_num,len(self.freq)))

		for i in range (0,len(self.freq)):
		    z=(1420/self.freq[i])-1
		    print(z)
		    x = cosmo_params.cosmo.comoving_distance(z).value
		    size = round(x*angle_r)
		    #find coeval cube of round(z) and take freq[i] slice
		    z_ = round(z, 0)
		    for j in range(0,len(self.redshift)):
                        # if self.redshift[j] == round(z*2)/2:
                        if abs(self.redshift[j] - z_) < 1e-6:
			    #take slice at freq[i]
                            print("found box!")
                            x_min = cosmo_params.cosmo.comoving_distance(self.redshift[j]).value
                            k = (x-x_min)/delta_x
                            k=round(k)
                            print('k before', k)
                            # while (k >= cell_num) or (k < -cell_num):
			    #     #print('using cyclic boundary conditions')
                            #     k= k-cell_num
                            k %= cell_num
                            print('k after', k)
                            
                            add = np.array([str(round(self.freq[i],2)),str(round(z,2)),str(self.redshift[j]),str(round(k,2))])
                            write_to_file = np.vstack((write_to_file,add))
                            #print(redshift[j])
                            x_box = cosmo_params.cosmo.comoving_distance(self.redshift[j]).value
                            new_cell_num =round((x_box/x_max)*cell_num)
                            cell_diff = round((cell_num-new_cell_num)/2)
                            cell_diff_max = round(new_cell_num+cell_diff)
                            print('new cell num: {0}, cell diff: {1}, cell diff max: {2}'.format(new_cell_num, cell_diff, cell_diff_max))

			    #temporary array to cut coeval box at the correct slice and angular size
                            temp = self.Tb[j][cell_diff:cell_diff_max,cell_diff:cell_diff_max,k]
			    #interpolate up 
                            x = np.linspace(0, 1, new_cell_num)
                            y = np.linspace(0, 1, new_cell_num)
                            f = interpolate.interp2d(y, x, temp)
                            x2 = np.linspace(0, 1, cell_num)
                            y2 = np.linspace(0, 1, cell_num)

			    #scalling = round(new_cell_num)/cell_num
                            lc[:,:,i]=f(y2, x2)
			    #fits.writeto('lc_freq_slices/521_1000Mpc_'+str(freq[i])+'MHz.fits', lc[:,:,i],overwrite=True) uncomment if need each individual slice in a folder - used for OSKAR

                        else:
                            continue
                            
		return lc , write_to_file
                        
	def savefiles(self, lc, write_to_file):
                lc=np.array(lc)

                np.savetxt('test.txt',write_to_file, fmt='%s')
                filename ='lc_{0:.2g}MPc_HIIDIM{1:.2g}_Z{2}-{3}_step:{4}_freq:{5}-{6}_step:{7}.fits'.format(self.BOX_LEN, self.HII_DIM, self.z_start, self.n_z*self.z_step+self.z_start, self.z_step, self.freq_start, self.nfreq*self.freq_step+self.freq_start, self.freq_step)
                fits.writeto(filename,lc,overwrite=True)

	def run_code(self):
                self.initial_conditions()
                self.make_cubes()
                lc , write_to_file = self.make_lc()
                np.savetxt('test.txt',write_to_file, fmt='%s') # uncomment to check lc is building correctly 
                #self.savefiles(lc, write_to_file)

                return lc, self.freq

def main():
        do_a_thing = lightcone(100, 100, 6, 22, 1, 5, 50,150,1)
        do_a_thing.run_code()      

if __name__ == '__main__':
    main()

