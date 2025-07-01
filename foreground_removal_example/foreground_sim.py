import numpy as np
import configparser
#from tqdm.autonotebook import tqdm
import scipy.interpolate
import scipy.ndimage
import scipy.fft as fft
from scipy import signal
import pylab as pl


class ForeGsim:
    def __init__(self, configFile="", ngrid=256, imgsize=10, nfreq=1, numin=150.0,
                 nustep=0.5, nexgal=13640809, sync_pli=-2.5, ff_pli=-3.,
                 exgal_pli=-0.7, psf_n_pixel = [10,10],psf_fwhm = [1,1],
                 version = ""):
        """
        Initialises an instance of the ForeGsym class.

        Params
        -------
        configFile : text
            A file containing all the parameters to pass to the class.
        imgsize : int
            The size of the output image in degrees.
        ngrid : int
            The size of the grid to simulate.
        nfreq : int
            The total number of frequency to generate sims for.
        numin : float
            The minimum frequency at which to generate a sim.
        nustep : float
            The jump between ajacent frequencies.
        nexgal : int
            The number of extragalactic galaxies.
        sync_pli : float
            The powerlaw index for the synchrotron distribution.
        ff_pli : float
            The powerlaw index for the free-free distribution.
        exgal_pli : float
            The powerlaw index for the extragalactic distribution.
        gauss_n_pixel : int 2D list
            The pixel size of the 2D Gaussian.
        gauss_fwhm : int 2D list
            The FWHM of the Gaussian along each axis.
        version : str
            empty or "original". If empty, it will use latest version, if 'vibor', it will use the original code

        """

        if configFile:
            try:
                config = configparser.ConfigParser()
                config.read(configFile)

            except IOError as io:
                print("An error occured trying to read the configFile.")
                # logger.exception("An error occured trying to read the configFile.")
        else:

            self.ngrid = ngrid
            self.imgsize = imgsize

            self.nfreq = nfreq
            self.numin = numin
            self.nustep = nustep
            self.nu = numin + nustep * np.arange(nfreq)

            self.nexgal = nexgal

            self.exgal_pli = exgal_pli
            self.sync_pli = sync_pli
            self.ff_pli = ff_pli

            self.psf_n_pixel = psf_n_pixel
            self.psf_fwhm = psf_fwhm

            self.version = version



    def _gauss3d(self, pli, test=False, ran_field=None):
        """
        Generates a 3D random Gaussian field

        Params
        ------

        pli : float
            Powerlaw index for random field.
        test : boolean
            Changes between testing and production modes.
        ran_field : array
            Predefined random field used for testing.

        Returns
        -------
        array
        """


        n = self.ngrid

        kx = np.zeros([n])
        kx[0 : int(n / 2) + 1] = np.arange(int(n / 2) + 1)
        temp = np.arange(int(n / 2) - 1) + 1
        kx[int(n / 2) + 1 :] = -temp[::-1]

        kz = ky = kx

        if test:
            if ran_field is None:

                return None
            else:
                phi = 2 * np.pi * ran_field

        else:
            phi = 2 * np.pi * np.random.rand(n, n, n)

        if self.version == "original":
            kamp = np.zeros([n, n, n])
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        kamp[i, j, k] = np.sqrt(kx[i] ** 2 + ky[j] ** 2 + kz[k] ** 2)

            kamp[np.where(kamp == 0)] = 0.0000001

            amp = kamp ** pli



            FTmapR = np.sqrt(amp) * np.cos(phi)
            FTmapI = np.sqrt(amp) * np.sin(phi)
            for i in range(1, int(n / 2)):
                for j in range(1, n):
                    for k in range(1, n):
                        FTmapR[n - i, n - j, n - k] = FTmapR[i, j, k]
                        FTmapI[n - i, n - j, n - k] = FTmapI[i, j, k]

            # %%time
            i1 = 0
            i2 = int(n / 2)
            for j in range(1, int(n / 2)):
                for k in range(1, n):
                    FTmapR[i1, n - j, n - k] = FTmapR[i1, j, k]
                    FTmapI[i1, n - j, n - k] = -FTmapI[i1, j, k]
                    FTmapR[i2, n - j, n - k] = FTmapR[i2, j, k]
                    FTmapI[i2, n - j, n - k] = -FTmapI[i2, j, k]

            j1 = 0
            j2 = int(n / 2)
            for i in range(1, int(n / 2)):
                for k in range(1, n):
                    FTmapR[n - i, j1, n - k] = FTmapR[i, j1, k]
                    FTmapI[n - i, j1, n - k] = -FTmapI[i, j1, k]
                    FTmapR[n - i, j2, n - k] = FTmapR[i, j2, k]
                    FTmapI[n - i, j2, n - k] = -FTmapI[i, j2, k]

            k1 = 0
            k2 = int(n / 2)
            for j in range(1, int(n / 2)):
                for i in range(1, n):
                    FTmapR[n - i, n - j, k1] = FTmapR[i, j, k1]
                    FTmapI[n - i, n - j, k1] = -FTmapI[i, j, k1]
                    FTmapR[n - i, n - j, k2] = FTmapR[i, j, k2]
                    FTmapI[n - i, n - j, k2] = -FTmapI[i, j, k2]

            i1 = 0
            j1 = 0
            i2 = int(n / 2)
            j2 = int(n / 2)
            i3 = 0
            j3 = int(n / 2)
            i4 = int(n / 2)
            j4 = 0
            for k in range(1, int(n / 2)):
                FTmapR[i1, j1, n - k] = FTmapR[i1, j1, k]
                FTmapI[i1, j1, n - k] = -FTmapI[i1, j1, k]
                FTmapR[i2, j2, n - k] = FTmapR[i2, j2, k]
                FTmapI[i2, j2, n - k] = -FTmapI[i2, j2, k]
                FTmapR[i3, j3, n - k] = FTmapR[i3, j3, k]
                FTmapI[i3, j3, n - k] = -FTmapI[i3, j3, k]
                FTmapR[i4, j4, n - k] = FTmapR[i4, j4, k]
                FTmapI[i4, j4, n - k] = -FTmapI[i4, j4, k]

            k1 = 0
            j1 = 0
            k2 = int(n / 2)
            j2 = int(n / 2)
            k3 = 0
            j3 = int(n / 2)
            k4 = int(n / 2)
            j4 = 0
            for i in range(1, int(n / 2)):
                FTmapR[n - i, j1, k1] = FTmapR[i, j1, k1]
                FTmapI[n - i, j1, k1] = -FTmapI[i, j1, k1]
                FTmapR[n - i, j2, k2] = FTmapR[i, j2, k2]
                FTmapI[n - i, j2, k2] = -FTmapI[i, j2, k2]
                FTmapR[n - i, j3, k3] = FTmapR[i, j3, k3]
                FTmapI[n - i, j3, k3] = -FTmapI[i, j3, k3]
                FTmapR[n - i, j4, k4] = FTmapR[i, j4, k4]
                FTmapI[n - i, j4, k4] = -FTmapI[i, j4, k4]

            i1 = 0
            k1 = 0
            i2 = int(n / 2)
            k2 = int(n / 2)
            i3 = 0
            k3 = int(n / 2)
            i4 = int(n / 2)
            k4 = 0
            for j in range(1, int(n / 2)):
                FTmapR[i1, n - j, k1] = FTmapR[i1, j, k1]
                FTmapI[i1, n - j, k1] = -FTmapI[i1, j, k1]
                FTmapR[i2, n - j, k2] = FTmapR[i2, j, k2]
                FTmapI[i2, n - j, k2] = -FTmapI[i2, j, k2]
                FTmapR[i3, n - j, k3] = FTmapR[i3, j, k3]
                FTmapI[i3, n - j, k3] = -FTmapI[i3, j, k3]
                FTmapR[i4, n - j, k4] = FTmapR[i4, j, k4]
                FTmapI[i4, n - j, k4] = -FTmapI[i4, j, k4]

            FTmapR[0, 0, 0] = 0.000001
            FTmapI[0, 0, 0] = 0.0

            FTmapI[int(n / 2), 0, 0] = 0.0
            FTmapI[0, int(n / 2), 0] = 0.0
            FTmapI[0, 0, int(n / 2)] = 0.0
            FTmapI[int(n / 2), int(n / 2), 0] = 0.0
            FTmapI[0, int(n / 2), int(n / 2)] = 0.0
            FTmapI[int(n / 2), 0, int(n / 2)] = 0.0
            FTmapI[int(n / 2), int(n / 2), int(n / 2)] = 0.0

            FTmap = FTmapR + 1j * FTmapI

            IFTmap = fft.ifftn(FTmap)

            Rmap = np.real(IFTmap)
            Rmap = Rmap / np.std(Rmap)



        elif self.version == "":
            kamp = np.zeros([n, n, n])
            i,j,k = np.meshgrid(range(n),range(n),range(n))
            kamp[i,j,k] = np.sqrt(kx[i]**2 + ky[j]**2 + kz[k]**2)
            kamp[np.where(kamp == 0)] = 0.0000001
            amp = kamp ** pli

            FTmapR = np.sqrt(amp) * np.cos(phi)
            FTmapI = np.sqrt(amp) * np.sin(phi)
            i,j,k = np.meshgrid(range(1, int(n / 2)),range(1,n),range(1,n))
            FTmapR[n - i, n - j, n - k] = FTmapR[i, j, k]
            FTmapI[n - i, n - j, n - k] = FTmapI[i, j, k]


            i,j,k = np.meshgrid([0,int(n / 2)],range(1, int(n / 2)),range(1,n))
            FTmapR[i, n - j, n - k] = FTmapR[i, j, k]
            FTmapI[i, n - j, n - k] = -FTmapI[i, j, k]

            i,j,k = np.meshgrid(range(1, int(n / 2)),[0,int(n / 2)],range(1,n))
            FTmapR[n - i, j, n - k] = FTmapR[i, j, k]
            FTmapI[n - i, j, n - k] = -FTmapI[i, j, k]

            i,j,k = np.meshgrid(range(1,n),range(1, int(n / 2)),[0,int(n / 2)])
            FTmapR[n - i, n - j, k] = FTmapR[i, j, k]
            FTmapI[n - i, n - j, k] = -FTmapI[i, j, k]

            ilist = [0,int(n / 2),0,int(n / 2)]
            jlist = [0,int(n / 2),int(n / 2),0]
            i,j,k = np.meshgrid(ilist,jlist,range(1, int(n / 2)))
            FTmapR[i, j, n - k] = FTmapR[i, j, k]
            FTmapI[i, j, n - k] = -FTmapI[i, j, k]


            jlist = [0,int(n / 2),int(n / 2),0]
            klist = [0,int(n / 2),0,int(n / 2)]
            i,j,k = np.meshgrid(range(1, int(n / 2)),jlist,klist)
            FTmapR[n - i, j, k] = FTmapR[i, j, k]
            FTmapI[n - i, j, k] = -FTmapI[i, j, k]

            ilist = [0,int(n / 2),0,int(n / 2)]
            klist = [0,int(n / 2),int(n / 2),0]
            i,j,k = np.meshgrid(ilist,range(1, int(n / 2)),klist)
            FTmapR[i, n - j, k] = FTmapR[i, j, k]
            FTmapI[i, n - j, k] = -FTmapI[i, j, k]


            FTmapR[0, 0, 0] = 0.000001
            FTmapI[0, 0, 0] = 0.0

            FTmapI[int(n / 2), 0, 0] = 0.0
            FTmapI[0, int(n / 2), 0] = 0.0
            FTmapI[0, 0, int(n / 2)] = 0.0
            FTmapI[int(n / 2), int(n / 2), 0] = 0.0
            FTmapI[0, int(n / 2), int(n / 2)] = 0.0
            FTmapI[int(n / 2), 0, int(n / 2)] = 0.0
            FTmapI[int(n / 2), int(n / 2), int(n / 2)] = 0.0

            FTmap = FTmapR + 1j * FTmapI

            IFTmap = fft.ifftn(FTmap)

            Rmap = np.real(IFTmap)
            Rmap = Rmap / np.std(Rmap)

        else:
            raise Exception('Unknown version given')



        # The transpose is to much the format of the output of the original IDL code.
        return Rmap.T

    def _random_powerLaw(self, power, x_range=[0, 1], size=1):
        """
        Power-law gen for pdf(x) propto x^{g-1} for a<=x<=b

        Params
        ------

        power : float
        x_range : list
        size : int

        Returns
        -------
        array
        """
        a, b = x_range
        g = power + 1
        r = np.random.random(size=size)
        ag, bg = a ** g, b ** g
        return (ag + (bg - ag) * r) ** (1.0 / g)

    def _gauss2d_psf(self, n_pixel=[10, 10], FWHM=[1, 1]):
        """
        Normalised 2D Gaussian

        Params
        ------

        n_pixel : int 2D list
            The pixel size of the 2D Gaussian.
        FWHM : int 2D list
            The FWHM of the Gaussian along each axis.

        Returns
        -------
        array

        """

        nx_pix = n_pixel[0]
        ny_pix = n_pixel[1]

        sigma_x = FWHM[0] / (2 * np.sqrt(2 * np.log(2)))
        sigma_y = FWHM[1] / (2 * np.sqrt(2 * np.log(2)))

        x = np.linspace(-nx_pix / 2, nx_pix / 2, nx_pix)
        y = np.linspace(-ny_pix / 2, ny_pix / 2, ny_pix)

        xx, yy = np.meshgrid(x, y)

        g = np.exp(-((xx ** 2) / (2 * sigma_x ** 2) + (yy ** 2) / (2 * sigma_y ** 2)))

        return g / np.sum(g)

    def _congrid(self, a, newdims, method="linear", centre=False, minusone=False):
        """
        Taken from scipy recipe book:

        Arbitrary resampling of source array to new dimension sizes.
        Currently only supports maintaining the same number of dimensions.
        To use 1-D arrays, first promote them to shape (x,1).

        Uses the same parameters and creates the same co-ordinate lookup points
        as IDL''s congrid routine, which apparently originally came from a VAX/VMS
        routine of the same name.

        method:
        neighbour - closest value from original data
        nearest and linear - uses n x 1-D interpolations using
                             scipy.interpolate.interp1d
        (see Numerical Recipes for validity of use of n 1-D interpolations)
        spline - uses ndimage.map_coordinates

        centre:
        True - interpolation points are at the centres of the bins
        False - points are at the front edge of the bin


        minusone:
        For example- inarray.shape = (i,j) & new dimensions = (x,y)
        False - inarray is resampled by factors of (i/x) * (j/y)
        True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
        This prevents extrapolation one element beyond bounds of input array.
        """
        if not a.dtype in [np.float64, np.float32]:
            a = a.astype(float)

        m1 = np.cast[int](minusone)
        ofs = np.cast[int](centre) * 0.5
        old = np.array(a.shape)
        ndims = len(a.shape)
        if len(newdims) != ndims:
            print(
                "[congrid] dimensions error. "
                "This routine currently only support "
                "rebinning to the same number of dimensions."
            )
            return None
        newdims = np.asarray(newdims, dtype=float)
        dimlist = []

        if method == "neighbour":
            for i in range(ndims):
                base = np.indices(newdims)[i]
                dimlist.append((old[i] - m1) / (newdims[i] - m1) * (base + ofs) - ofs)
            cd = np.array(dimlist).round().astype(int)
            newa = a[list(cd)]
            return newa

        elif method in ["nearest", "linear"]:
            # calculate new dims
            for i in range(ndims):
                base = np.arange(newdims[i])
                dimlist.append((old[i] - m1) / (newdims[i] - m1) * (base + ofs) - ofs)
            # specify old dims
            olddims = [np.arange(i, dtype=np.float) for i in list(a.shape)]

            # first interpolation - for ndims = any
            mint = scipy.interpolate.interp1d(olddims[-1], a, kind=method)
            newa = mint(dimlist[-1])

            trorder = [ndims - 1] + [*range(ndims - 1)]
            for i in range(ndims - 2, -1, -1):
                newa = newa.transpose(trorder)

                mint = scipy.interpolate.interp1d(olddims[i], newa, kind=method)
                newa = mint(dimlist[i])

            if ndims > 1:
                # need one more transpose to return to original dimensions
                newa = newa.transpose(trorder)

            return newa
        elif method in ["spline"]:
            oslices = [slice(0, j) for j in old]
            oldcoords = np.ogrid[oslices]
            nslices = [slice(0, j) for j in list(newdims)]
            newcoords = np.mgrid[nslices]

            newcoords_dims = range(np.rank(newcoords))
            # make first index last
            newcoords_dims.append(newcoords_dims.pop(0))
            newcoords_tr = newcoords.transpose(newcoords_dims)
            # makes a view that affects newcoords

            newcoords_tr += ofs

            deltas = (np.asarray(old) - m1) / (newdims - m1)
            newcoords_tr *= deltas

            newcoords_tr -= ofs

            newa = scipy.ndimage.map_coordinates(a, newcoords)
            return newa
        else:
            print(
                "Congrid error: Unrecognized interpolation type.\n",
                "Currently only 'neighbour', 'nearest','linear',",
                "and 'spline' are supported.",
            )
            return None

    def _rebin2D(self, a, new_shape=[None, None], mean=True):
        """
        Reduces the number of dimensions of an array through rebining.

        Params
        ------

        a : array
            Original array to be changed
        new_shape : list
            Dimensions of new array. Has to be a multiple of original array size.
        mean : boolean
            Changes the output array from a regular sum to mean.

        Returns
        -------
        array

        """

        new_shape = np.array(new_shape)
        factor_tmp = a.shape / new_shape
        rebin_factor = factor_tmp.astype(int)
        check = factor_tmp == rebin_factor
        if check[0] and check[1]:
            if mean:
                b = a.reshape(
                    new_shape[0], rebin_factor[0], new_shape[1], rebin_factor[1]
                )  # reshape into 4D array
                b = b.mean(1).mean(
                    2
                )  # Take mean across 1 axis to give a 3D array, then take mean accros last axis to get 2D array

            else:
                b = a.reshape(
                    new_shape[0], rebin_factor[0], new_shape[1], rebin_factor[1]
                )  # reshape into 4D array
                b = b.sum(1).sum(
                    2
                )  # Take sum across 1 axis to give a 3D array, then take sum accros last axis to get 2D arra

        else:
            print("New shape is not a multiple of old shape.")
            b = None

        return b

    def read_IDL_file(self, fname, shape=(256, 256, 256)):
        """
        Written specifically to read in IDL output arrays

        Params
        ------
        fname : str
            IDL file name
        shape : tuple
            Shape of array in IDL file.

        Returns
        -------
        array
        """
        file = open(fname, "r")
        lines = file.readlines()

        # remove front space and \n character
        lines = [lines[i][:-2].strip() for i in range(len(lines))]

        flatten_lines = np.array(lines).flatten()

        # separate strings into list and join all lists together into one array
        stacked_lines = np.hstack(np.char.split(flatten_lines, sep="  "))

        # remove elements of array with empty strings
        cleaned_lines = stacked_lines[stacked_lines != ""].astype(float)

        return cleaned_lines.reshape(shape)

    def gen_sync_map(self, test=False, ran_field_asyn=None, ran_field_plisyn=None):
        """
        Generates a synchrotron emmission foreground map.

        Params
        ------

        test : boolean
            Changes between testing and production modes.
        ran_field_asyn : array
            Predefined random field used for testing.
        ran_field_plisyn : array
            Predefined random field used for testing.

        Returns
        -------
        array

        """
        pli = self.sync_pli

        Asyn = self._gauss3d(pli=pli, test=test, ran_field=ran_field_asyn)
        PLIsyn = self._gauss3d(pli=pli, test=test, ran_field=ran_field_plisyn)
        norm = np.std(np.sum(Asyn, 2))
        Asyn = Asyn * 3.0 / norm  # @150MHz
        PLIsyn = -2.55 + PLIsyn * 0.1

        syn = np.zeros((self.ngrid, self.ngrid, self.nfreq))
        for i in range(self.nfreq):
            syntemp = Asyn * (self.nu[i] / 150.0) ** PLIsyn
            syn[:, :, i] = np.sum(syntemp, 2).T
            # Transpose to much orginal IDL output format
            print(("SYN @" + str(self.nu[i]) + "MHz: DONE!").split())

        return syn

    def gen_freefree_map(self, test=False, ran_field_aff=None, ran_field_pliff=None):
        """
        Generates a free-free emmission foreground map.

        Params
        ------

        test : boolean
            Changes between testing and production modes.
        ran_field_aff : array
            Predefined random field used for testing.
        ran_field_pliff : array
            Predefined random field used for testing.

        Returns
        -------
        array

        """

        pli = self.ff_pli

        Aff = self._gauss3d(pli=pli, test=test, ran_field=ran_field_aff)
        PLIff = self._gauss3d(pli=pli, test=test, ran_field=ran_field_pliff)

        norm = np.std(np.sum(Aff, 2))
        Aff = Aff * 0.03 / norm  # @150MHz
        PLIff = -2.15 + PLIff * 0.05

        ff = np.zeros((self.ngrid, self.ngrid, self.nfreq))
        for i in range(self.nfreq):
            fftemp = Aff * (self.nu[i] / 150.0) ** PLIff
            ff[:, :, i] = np.sum(
                fftemp, 2
            ).T  # Transpose to much orginal IDL output format
            print(("FF @" + str(self.nu[i]) + "MHz: DONE!").split())

        return ff

    def gen_exgal_map(self, test = False, test_phi = None, test_dist = None):
        """
        Generates an extragalactic foreground map.

        Params
        ------

        test : boolean
            Changes between testing and production modes.
        test_phi : array
            Predefined random field used for testing.
        test_dist : array
            Predefined random field used for testing.

        Returns
        -------
        array

        """
        nexgal = self.nexgal
        imgsize = self.imgsize
        exgal_pli = self.exgal_pli

        if test == True:
            phi = test_phi
            dist = test_dist
        else:
            phi = 2 * np.pi * np.random.rand(nexgal)
            dist = self._random_powerLaw(power=exgal_pli, x_range=[0.005, imgsize], size=nexgal)

        x = np.zeros(nexgal)
        y = np.zeros(nexgal)

        for i in range(1, nexgal):
            x[i] = x[i - 1] + dist[i] * np.cos(phi[i])
            y[i] = y[i - 1] + dist[i] * np.sin(phi[i])

            if x[i] > imgsize / 2:
                x[i] = x[i] - imgsize
            elif x[i] < -imgsize / 2:
                x[i] = x[i] + imgsize
            if y[i] > imgsize / 2:
                y[i] = y[i] - imgsize
            elif y[i] < -imgsize / 2:
                y[i] = y[i] + imgsize

        ngridtmp = 16384
        res = imgsize / (ngridtmp - 1)

        xtmp = np.round((x + imgsize / 2) / res).astype(int)
        ytmp = np.round((y + imgsize / 2) / res).astype(int)

        n_pixel = self.psf_n_pixel
        FWHM = self.psf_fwhm

        psf = self._gauss2d_psf(n_pixel=n_pixel, FWHM=FWHM)

        bmaj = 2.0 * 60.0
        bmin = 2.0 * 60.0

        c = 2.99792458 * 1e8
        kb = 1.3806503 * 1e-23
        arcsectorad = np.pi / 648000.0
        solidang = (np.pi * (bmaj * arcsectorad) * (bmin * arcsectorad) / (4.0 * np.log(2.0)))

        exgal_data = np.loadtxt("exgalFG.1muJy.unr", usecols=(2, 3))

        extgalSJy = exgal_data[:, 0]
        extgalSpli = exgal_data[:, 1]

        exgal = np.zeros((self.ngrid, self.ngrid, self.nfreq))

        for j in range(self.nfreq):
            imgtmp = np.zeros((ngridtmp, ngridtmp))
            imgtmp[xtmp, ytmp] = extgalSJy * (self.nu[j] / 151.0) ** (extgalSpli)
            #Flux cut
            imgtmp[np.where(imgtmp > 0.05 )] = 0

            imgtmpnu = self._rebin2D(a=imgtmp, new_shape=[2048, 2048],mean = False)

            imgtmpnuC = signal.convolve2d(imgtmpnu, psf,mode='same')
            print(imgtmpnuC.shape)
            #imgnu = (self._congrid(a=imgtmpnuC, newdims=(self.ngrid, self.ngrid))* (16384 / self.ngrid) ** 2)
            imgnu = self._rebin2D(a = imgtmpnuC,new_shape = [self.ngrid,self.ngrid],mean = False)
            
            exgal[:, :, j] = ((imgnu * 1e-26) / solidang)* c ** 2 / (2.0 * kb * (self.nu[j] * 1e6) ** 2)

        return exgal
