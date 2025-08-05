# imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py                       # For reading HDF5 data.
import sys
import os


# Function to get snapshot filename. --> adjust function if necessary based on standard file names
def get_fname_snap(i, snapdir, verbose=True):
    fname = os.path.join(snapdir, 'snapshot_{0:03d}.hdf5'.format(i))
    if verbose:
        print('filename: {0:s}'.format(fname))
    return fname



def dust_density (gas_density, dust_mass = False, m_unit = 'grams', figure_size_x = 1, figure_size_y = 1, figure_size_z = 1, 
                 unit = 'AU', y_unit = 'AU', z_unit = 'AU', x_scale = 0, y_scale = 0, z_scale = 0):
    """ Calculates dust mass and dust density from gas density.
     Uses the 1% dust to gas ratio to calculate dust density.
    If dust_mass = True, calculates dust mass as well.
   
    Parameters
    ----------
        density: 2D numpy array
            x and y density values in g cm^-3
           
        m_unit: string
            choice of input mass units based on density units. automatically set to 'grams' (accepts "kg" and "solar")
           
        figure_size_x, figure_size_y, figure_size_z: float
            size of image (in inches) in a given direction. automatically set to 1 in all directions
            if working in two dimensions, leave figure_size_z = 1
       
        x_unit, y_unit, z_unit: string
            choice of dimensional units. automatically set to 'AU' (accepts "meters", "pc", and "kpc")
           
        x_scale, y_scale, z_scale: float
            length of each dimension in chosen unit
           
    Returns
    -------
        dust density: 2D numpy array
            dust densities (in g cm^-3) and respective calculated dust masses (in g) if selected
    """
    
    import numpy as np
    
    # Calculating dust density
    dust_density = gas_density * 0.01

    # Conditional dust mass
    if dust_mass: 
        if x_scale == 0 and y_scale == 0:
            raise ValueError("x_scale or y_scale must be set to nonzero values.")
    
        # Unit conversions
        def unit_to_cm(unit, scale):
            if unit == 'AU':
                return scale * 1.496e+13
            elif unit == 'meters':
                return scale * 100
            elif unit == 'pc':
                return scale * 3.086e+18
            elif unit == 'kpc':
                return scale * 3.086e+21
            else:
                raise ValueError(f"Unknown unit {unit}")
    
        x_scale_cm = unit_to_cm(x_unit, x_scale) if x_scale != 1 else 0
        y_scale_cm = unit_to_cm(y_unit, y_scale) if y_scale != 1 else 0
        z_scale_cm = unit_to_cm(z_unit, z_scale) if z_scale != 1 else 0
        
        if z_scale_cm == 0:
            # Dimensions
            y_pix, x_pix = density.shape  # note shape order (y, x) since 2D array
            z_pix = 0
            # Pixel size in cm
            x_pixel = x_scale_cm / x_pix
            y_pixel = y_scale_cm / y_pix
            z_pixel = 0
            A_pixel = x_pixel * y_pixel
        elif x_scale_cm == 0:
            # Dimensions
            y_pix, z_pix = density.shape  # note shape order (y, z) since 2D array
            x_pix = 0
            # Pixel size in cm
            z_pixel = z_scale_cm / z_pix
            y_pixel = y_scale_cm / y_pix
            x_pixel = 0
            A_pixel = z_pixel * y_pixel
        elif y_scale_cm == 0:
            # Dimensions
            x_pix, z_pix = density.shape  # note shape order (x, z) since 2D array
            y_pix = 0
            # Pixel size in cm
            z_pixel = z_scale_cm / z_pix
            x_pixel = x_scale_cm / x_pix
            y_pixel = 0
            A_pixel = z_pixel * x_pixel
    
        print(f"x_scale_cm = {x_scale_cm:.3e} cm, y_scale_cm = {y_scale_cm:.3e} cm, z_scale_cm = {z_scale_cm:.3e} cm" )
        print(f"Image pixels: x_pix = {x_pix}, y_pix = {y_pix}, z_pix = {z_pix}")
        print(f"Pixel size: x_pixel = {x_pixel:.3e} cm, y_pixel = {y_pixel:.3e} cm, z_pixel = {z_pixel:.3e} cm")
        print(f"Pixel area A_pixel = {A_pixel:.3e} cm^2")

        # Calculate masses from surface density (density in g/cm^2)
        m_gas = density * A_pixel
        m_dust = m_gas * 0.01

        # Array with dust density and mass
        dust_density_mass = np.stack([dust_density, m_dust], axis = -1)
        return dust_density_mass 

    else: 
        return dust_density




def flux_density(density, T, f, m_unit = 'grams', figure_size_x = 1, figure_size_y = 1, figure_size_z = 1, 
                 x_unit = 'AU', y_unit = 'AU', z_unit = 'AU', x_scale = 0, y_scale = 0, z_scale = 0):

    """ Calculates the flux density from a gas density projection of an optically think disk.
    First computes gas mass by dividing density by pixel area/volume.
    Then uses the 1% dust to gas mass ratio to calculate dust mass.
    Finally, computes the flux density from the dust mass.
   
    Parameters
    ----------
        density: 2D numpy array
            x and y density values in g cm^-3
       
        T: float
            temperature in K
       
        f: float
            observational frequency in GHz
           
        m_unit: string
            choice of input mass units based on density units. automatically set to 'grams' (accepts "kg" and "solar")
           
        figure_size_x, figure_size_y, figure_size_z: float
            size of image (in inches) in a given direction. automatically set to 1 in all directions
            if working in two dimensions, leave figure_size_z = 1
       
        x_unit, y_unit, z_unit: string
            choice of dimensional units. automatically set to 'AU' (accepts "meters", "pc", and "kpc")
           
        x_scale, y_scale, z_scale: float
            length of each dimension in chosen unit
           
    Returns
    -------
        flux density: 2D numpy array
            flux densities (in mJy) and respective gas densities(in g cm^-2) and calculated dust masses (in g)
   
    Example
    """
    
    import numpy as np

    if x_scale == 0 and y_scale == 0:
        raise ValueError("x_scale or y_scale must be set to nonzero values.")

    # Unit conversions
    def unit_to_cm(unit, scale):
        if unit == 'AU':
            return scale * 1.496e+13
        elif unit == 'meters':
            return scale * 100
        elif unit == 'pc':
            return scale * 3.086e+18
        elif unit == 'kpc':
            return scale * 3.086e+21
        else:
            raise ValueError(f"Unknown unit {unit}")

    x_scale_cm = unit_to_cm(x_unit, x_scale) if x_scale != 0 else 0
    y_scale_cm = unit_to_cm(y_unit, y_scale) if y_scale != 0 else 0
    z_scale_cm = unit_to_cm(z_unit, z_scale) if z_scale != 0 else 0


    if z_scale_cm == 0:
        # Dimensions
        y_pix, x_pix = density.shape  # note shape order (y, x) since 2D array
        z_pix = 0
        # Pixel size in cm
        x_pixel = x_scale_cm / x_pix
        y_pixel = y_scale_cm / y_pix
        z_pixel = 0
        A_pixel = x_pixel * y_pixel
    elif x_scale_cm == 0:
        # Dimensions
        y_pix, z_pix = density.shape  # note shape order (y, z) since 2D array
        x_pix = 0
        # Pixel size in cm
        z_pixel = z_scale_cm / z_pix
        y_pixel = y_scale_cm / y_pix
        x_pixel = 0
        A_pixel = z_pixel * y_pixel
    elif y_scale_cm == 0:
        # Dimensions
        x_pix, z_pix = density.shape  # note shape order (x, z) since 2D array
        y_pix = 0
        # Pixel size in cm
        z_pixel = z_scale_cm / z_pix
        x_pixel = x_scale_cm / x_pix
        y_pixel = 0
        A_pixel = z_pixel * x_pixel

    print(f"x_scale_cm = {x_scale_cm:.3e} cm, y_scale_cm = {y_scale_cm:.3e} cm, z_scale_cm = {z_scale_cm:.3e} cm" )
    print(f"Image pixels: x_pix = {x_pix}, y_pix = {y_pix}, z_pix = {z_pix}")
    print(f"Pixel size: x_pixel = {x_pixel:.3e} cm, y_pixel = {y_pixel:.3e} cm, z_pixel = {z_pixel:.3e} cm")
    print(f"Pixel area A_pixel = {A_pixel:.3e} cm^2")

    # Calculate masses from surface density (density in g/cm^2)
    m_gas = density * A_pixel
    m_dust = m_gas * 0.01

    print(f"m_gas: min = {np.min(m_gas):.3e} g, max = {np.max(m_gas):.3e} g, mean = {np.mean(m_gas):.3e} g")
    print(f"m_dust: min = {np.min(m_dust):.3e} g, max = {np.max(m_dust):.3e} g, mean = {np.mean(m_dust):.3e} g")

    # Adjust dust mass units if needed
    if m_unit == 'kg':
        m_dust /= 1000  # convert g to kg
        print("Converted m_dust from grams to kilograms.")
    elif m_unit == 'solar':
        m_dust /= 1.989e+33  # convert g to solar masses
        print("Converted m_dust from grams to solar masses.")

    # Constants
    h = 6.626e-27  # Planck's constant in erg s
    c = 3.0e10     # Speed of light in cm/s
    k = 1.38e-16   # Boltzmann constant in erg/K

    # Planck function B_v calculation
    freq = f * 10e9  # GHz to Hz
    a = (2.0 * h * freq**3) / c**2
    b = (h * freq) / (k * T)
    B_v = a / (np.exp(b) - 1.0)

    print(f"Planck function B_v = {B_v:.3e} erg s^-1 cm^-2 Hz^-1 sr^-1")

    # Opacity and distance
    K_v = f / 100  # cm^2 g^-1, opacity at frequency f in GHz
    d_cm = 4.3204e+20  # distance in cm (140 pc)
    # omega = A_pixel / d_cm**2  # solid angle in steradians

    print(f"Opacity K_v = {K_v:.3e} cm^2 g^-1")
    print(f"Distance d_cm = {d_cm:.3e} cm")
    # print(f"Solid angle omega = {omega:.3e} sr")

    # Flux density calculation in erg s^-1 cm^-2 Hz^-1
    f_v = (m_dust * K_v * B_v) / ((d_cm)**2)
    # Convert to mJy
    F_v = f_v / 1e-26

    print(f"Flux density F_v: min = {np.min(F_v):.3e} mJy, max = {np.max(F_v):.3e} mJy, mean = {np.mean(F_v):.3e} mJy")

    # Stack results
    flux = np.stack([density, m_dust, F_v], axis=-1)

    return flux






def write_disk_model(flux_centered, save_dir, d_pc=140, x_scale=0, y_scale=0, z_scale=0, 
                     r=None, inc=None, pa=None, zoom=None, T=None, disk_name=None, ax=None,
                     output_prefix="diskmodel",
                     incenter_GHz=230.0,  # <-- frequency in GHz
                     ra_deg=270.0,        # <-- RA in degrees
                     dec_deg=-23.0):      # <-- DEC in degrees
    
    from astropy.io import fits
    import numpy as np
    import os
    from datetime import datetime

    # Load template header
    header_file = "/workspace/budamgunta/snapshots/ppdisk672_GHz_50pc_input.fits"
    header = fits.getheader(header_file)

    # Determine image size
    ny, nx = flux_centered.shape
    naxis1 = flux_centered.shape[1]
    naxis2 = flux_centered.shape[0]

    # Compute pixel scale
    if z_scale == 0:
        cdelt1 = (x_scale / (naxis1 * d_pc)) / 3600
        cdelt2 = (y_scale / (naxis2 * d_pc)) / 3600
    elif x_scale == 0:
        cdelt1 = (z_scale / (naxis1 * d_pc)) / 3600
        cdelt2 = (y_scale / (naxis2 * d_pc)) / 3600
    elif y_scale == 0:
        cdelt1 = (z_scale / (naxis1 * d_pc)) / 3600
        cdelt2 = (x_scale / (naxis2 * d_pc)) / 3600

    # Set header values
    # Update FITS header for 3D WCS
    header['NAXIS'] = 3
    header['NAXIS1'] = nx
    header['NAXIS2'] = ny
    header['NAXIS3'] = 1
    header['CRPIX1'] = nx // 2
    header['CRPIX2'] = ny // 2
    header['CRPIX3'] = 1
    header['CDELT1'] = -cdelt1
    header['CDELT2'] = cdelt2
    header['CDELT3'] = 1
    header['CRVAL1'] = ra_deg
    header['CRVAL2'] = dec_deg
    header['CRVAL3'] = 1  # Stokes I
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CTYPE3'] = 'STOKES'
    header['BUNIT'] = 'mJy/pixel'
    header['ORIGIN'] = 'CASA compatible FITS'
    header['DATE'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Clean up any existing 4D keys
    for key in ['NAXIS4', 'CTYPE4', 'CRPIX4', 'CRVAL4', 'CDELT4']:
        if key in header:
            del header[key]
    

    # Generate filename
    name_parts = [output_prefix]
    if disk_name: name_parts.append(disk_name)
    if r is not None: name_parts.append(f"r{r}")
    if inc is not None: name_parts.append(f"inc{inc}")
    if pa is not None: name_parts.append(f"PA{pa}")
    if zoom is not None: name_parts.append(f"zoom{zoom}")
    if T is not None: name_parts.append(f"T{T}K")
    if ax is not None: name_parts.append(f"{ax} axis")
    filename = "_".join(name_parts) + ".fits"

    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, filename)
    fits.writeto(full_path, flux_centered, header, overwrite=True)
    print(f"FITS file written successfully: {full_path}")
    return full_path