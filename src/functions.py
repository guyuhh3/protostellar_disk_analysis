# imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py                       # For reading HDF5 data.
import sys
import os

# importing radmc-3d
sys.path.append('/workspace/budamgunta/anaconda3/radmc3d-2.0-master/python/radmc3dPython')
import radmc3dPy


# Function to get snapshot filename. --> adjust function if necessary based on standard file names
def get_fname_snap(i, snapdir, verbose=True):
    fname = os.path.join(snapdir, 'snapshot_{0:03d}.hdf5'.format(i))
    if verbose:
        print('filename: {0:s}'.format(fname))
    return fname




# Example: function to load code units and store as Python dictionary.
def get_code_units_from_snap(i, snapdir, B_unit=1e4, verbose=True):
    '''
    Function to load code units and store as Python dictionary.

    Returns
    --------
        {'m_unit': np.float64(1.989e+33),
         'l_unit': np.float64(3.085678e+18),
         'v_unit': np.float64(100.0),
         't_unit': np.float64(3.085678e+16),
         'B_unit': 10000.0,
         'G_code': np.float64(4300.71057317063)}
         
    Example
    --------
        unit_dict = get_code_units_from_snap(i, snapdir, verbose=False)
        m_unit = unit_dict['m_unit']
        print(m_unit) --> 1.989e+33
    '''
    fname = get_fname_snap(i, snapdir, verbose=verbose)
    with h5py.File(fname, 'r') as f:
        header = f['Header']
        G_code = header.attrs['Gravitational_Constant_In_Code_Inits']
        l_unit = header.attrs['UnitLength_In_CGS']     # Typically 1 parsec.
        m_unit = header.attrs['UnitMass_In_CGS']       # Typically 1 Solar mass.
        v_unit = header.attrs['UnitVelocity_In_CGS']   # Typically 1 meter/second.
        t_unit = l_unit / v_unit
        t_unit_kyr = t_unit / (3600.0 * 24.0 * 365.0 * 1e3)
        t_unit_myr = t_unit / (3600.0 * 24.0 * 365.0 * 1e6)
    if verbose:
        print('unit length:   {0:.3e} cm'.format(l_unit))
        print('unit mass:     {0:.3e} g'.format(m_unit))
        print('unit velocity: {0:.1f} cm/s'.format(v_unit))
        print('unit time:     {0:.3e} s = {1:.2e} kyr = {2:.2f} Myr'.format(t_unit, t_unit_kyr, t_unit_myr))
        print('unit B:        {0:.3e} Gauss'.format(B_unit))
        print('grav. constant [code units]: {0:.3g}'.format(G_code))
    unit_dict = {'m_unit':m_unit,
                 'l_unit':l_unit,
                 'v_unit':v_unit,
                 't_unit':t_unit,
                 'B_unit':B_unit,
                 'G_code':G_code}
    return unit_dict






# Example: function to load PartType0 (gas), PartType5 (star) data and store in dict.
def load_snapshot_data(i, snapdir, verbose=False):
    """
        Function to load specific data as Python dictionary.
    
        Returns
        --------
            {'m_unit': np.float64(1.989e+33),
             'l_unit': np.float64(3.085678e+18),
             .
             .
             .
             'gas': {'ids': array([88842, 63114, 57352, ..., 91890, 98428, 82261],
            shape=(96784,), dtype=uint64),
      'm': array([1.e-05, 1.e-05, 1.e-05, ..., 1.e-05, 1.e-05, 1.e-05],
            shape=(96784,), dtype=float32),
      'rho': array([157232.89 , 221471.5  , 252874.39 , ...,  36395.195,  23362.99 ,
              77066.49 ], shape=(96784,), dtype=float32),
      'X': array([[0.04655657, 0.05399808, 0.0437885 ],
             [0.04680883, 0.05428582, 0.04394167],
             [0.04678837, 0.05413829, 0.04448474],
             ...,
             [0.05248024, 0.05249692, 0.05317758],
             [0.05259772, 0.05284877, 0.05384233],
             [0.0524778 , 0.05177811, 0.05348589]], shape=(96784, 3)),
      'V': array([[-207.92772, -527.6626 ,  184.59543],
             [-199.10222, -537.41736,  108.93194],
             [-207.39722, -560.3308 ,  115.73126],
             ...,
             [-291.44278, -396.26642, -415.93265],
             [-284.18015, -322.40732, -415.10028],
             [-279.4914 , -290.03348, -479.5435 ]],
            shape=(96784, 3), dtype=float32),
      'B': array([[ 3.5531789e-09, -2.5034801e-08,  2.2479968e-08],
             [ 4.6566031e-09, -2.6961359e-08,  2.1830539e-08],
             [ 8.8659053e-09, -2.7008072e-08,  2.0164245e-08],
             ...,
             [ 2.6416880e-09,  5.3667537e-09,  1.5301726e-08],
             [ 1.0423825e-09,  5.8965863e-09,  9.3616395e-09],
             [ 7.3809447e-10,  3.5460486e-09,  1.9489219e-08]],
            shape=(96784, 3), dtype=float32)},
     'stars': {'ids': array([ 1054, 61757, 15056], dtype=uint64),
      'm': array([0.18758999, 0.06352   , 0.01768   ], dtype=float32),
      'X': array([[0.05042241, 0.04937311, 0.05049972],
             [0.0504567 , 0.04938993, 0.05048939],
             [0.0503906 , 0.04945482, 0.05051317]]),
      'V': array([[  992.0339,  -619.7486,   612.9943],
             [-1827.4503,  2775.8313, -2779.4504],
             [-3525.8079,  -886.1629,   671.1458]], dtype=float32)}}
             
        Example
        --------
            i = 180
            snapshot_data = load_snapshot_data(i, snapdir, verbose=False)
            p0_rho.shape --> (96784,)
    """
     
    fname = get_fname_snap(i, snapdir, verbose=verbose)
    with h5py.File(fname, 'r') as f:
        # Header data:
        header = f['Header']
        G_code = header.attrs['Gravitational_Constant_In_Code_Inits']
        l_unit = header.attrs['UnitLength_In_CGS']     # Typically 1 parsec.
        m_unit = header.attrs['UnitMass_In_CGS']       # Typically 1 Solar mass.
        v_unit = header.attrs['UnitVelocity_In_CGS']   # Typically 1 meter/second.
        t_unit = l_unit / v_unit
        B_unit = 1e4 
        # More relevant timescales.
        t_unit_kyr = t_unit / (3600.0 * 24.0 * 365.0 * 1e3)
        t_unit_myr = t_unit / (3600.0 * 24.0 * 365.0 * 1e6)
        # PartType0 (gas) data:
        p0     = f['PartType0']
        p0_ids = p0['ParticleIDs'][()]         # Particle IDs; shape = (N_gas, )
        p0_m   = p0['Masses'][()]              # Masses.
        p0_rho = p0['Density'][()]             # Density.
        p0_X   = p0['Coordinates'][()]         # Coordinates; shape (N_gas, 3)
        p0_V   = p0['Velocities'][()]          # Velocities.
        p0_B   = p0['MagneticField'][()]       # Magnetic field.
        p0_T   = p0['Dust_Temperature'][()]    # Dust Temperature
        # PartType5 (star) data:
        stars_exist = False
        if 'PartType5' in f:
            stars_exist = True
            p5         = f['PartType5']
            p5_ids     = p5['ParticleIDs'][()]  # Particle IDs; shape = (N_stars, )
            p5_m       = p5['Masses'][()]       # Masses.
            p5_X       = p5['Coordinates'][()]  # Coordinates; shape = (N_stars, 3)
            p5_V       = p5['Velocities'][()]   # Velocities.
        # Save to dict.
        snapshot_data = {'m_unit':m_unit,
                         'l_unit':l_unit,
                         'v_unit':v_unit,
                         't_unit':t_unit,
                         'B_unit':B_unit,
                         'G_code':G_code,
                         'gas':{'ids':p0_ids,
                                'm':p0_m,
                                'rho':p0_rho,
                                'X':p0_X,
                                'V':p0_V,
                                'B':p0_B,
                                'T':p0_T}}
        if stars_exist:
            snapshot_data['stars'] = {'ids':p5_ids,
                                      'm': p5_m,
                                      'X':p5_X,
                                      'V':p5_V}
    return snapshot_data







def get_plot_data_from_YT(fname_snap, figdir, 
                          field_list=[('gas', 'density')], 
                          ax='x',
                          zoom=5,
                          center_sink_ids=None,
                          center_coords=None,
                          use_box=False,
                          weighted=False,
                          moment_field=False, verbose=True):
    
    import yt
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import h5py                       # For reading HDF5 data.
    import sys
    import os
    import unyt
    # Save YT plot data to Python dict.
    plot_data = {}
    
    # Default STARFORGE units.
    B_unit = 1e4        # Tesla in Gauss
    m_unit = 1.989e33   # solar mass in g
    l_unit = 3.086e+18  # parsec in cm
    v_unit = 100.0
    
    # Get new field data using YT.
    if verbose:
        print('Using YT to get new plot data...', flush=True)
    yt.set_log_level(50)
    unit_base = {'UnitMagneticField_in_gauss': B_unit,
                 'UnitLength_in_cm': l_unit,
                 'UnitMass_in_g': m_unit,
                 'UnitVelocity_in_cm_per_s': v_unit}
    ds = yt.load(fname_snap, unit_base=unit_base); ad = ds.all_data()
        
    # Can specify center_coords; else default to domain center.
    if center_coords is not None:
        c = ds.arr(center_coords, 'code_length')
    elif center_sink_ids is not None:
        # Load sink particle data
        with h5py.File(fname_snap, 'r') as f:
            p5 = f['PartType5']
            ids = p5['ParticleIDs'][()]
            coords = p5['Coordinates'][()]
            masses = p5['Masses'][()]
            idx = np.isin(ids, center_sink_ids)
            m = masses[idx]
            x = coords[idx, 0]
            y = coords[idx, 1]
            z = coords[idx, 2]
            cm_x = np.sum(m * x) / np.sum(m)
            cm_y = np.sum(m * y) / np.sum(m)
            cm_z = np.sum(m * z) / np.sum(m)
            c = ds.arr([cm_x, cm_y, cm_z], 'code_length')
    else:
        c = ds.domain_center

    
    # Get data region - use only the volume contained in this box for the projection.
    '''
    if use_box:
        box_size   = ds.domain_width.v[0]  # Should get simulation box size from domain dimensions.
        #box_size   = 0.1  # M1 calculations: R_cloud = 0.01 pc, R_box = 0.1 pc.
        half_width = ds.quan(box_size/(2.0 * zoom), 'code_length')
        left_edge  = c - half_width
        right_edge = c + half_width
        box        = ds.region(c, left_edge, right_edge, fields=field_list, ds=ds)
    '''
    
    # Get projection plot data.
    # Make a density-weighted projection of a field along a line of sight.
    if weighted:
        if use_box:
            prj = yt.ProjectionPlot(ds, ax, field_list, center=c, data_source=box,
                                    weight_field=('gas', 'density'))
        else:
            prj = yt.ProjectionPlot(ds, ax, field_list, center=c,
                                    weight_field=('gas', 'density'))
        
    # Make a weighted projection of the standard deviation of a field along a line of sight,
    # e.g., 1D line-of-sight velocity dispersion, ('gas', 'velocity_x').
    elif moment_field:
        if use_box:
            prj = yt.ProjectionPlot(ds, ax, new_field_list, center=c, data_source=box,
                                    weight_field=('gas', 'density'), moment=2)
        else:
            prj = yt.ProjectionPlot(ds, ax, new_field_list, center=c,
                                    weight_field=('gas', 'density'), moment=2)
    # Else just make an unweighted projection of a field along a line of sight.
    else:
        if use_box:
            prj = yt.ProjectionPlot(ds, ax, field_list, center=c, data_source=box)
        else:
            prj = yt.ProjectionPlot(ds, ax, field_list, center=c)
    prj.set_axes_unit('AU')
    prj.zoom(zoom)
    
    # Need to plot/save figures to save data.
    tempname = os.path.join(figdir, 'temp_prj.png')
    prj.save(tempname)
            
    plot_data_shape = (0, 0)
    for i, field_tuple in enumerate(list(prj.plots)):
        field_name = field_tuple[1]
        if weighted:
            field_name = '{0:s}_weighted'.format(field_name)
        if moment_field:
            field_name = '{0:s}_dispersion'.format(field_name)
        plot = prj.plots[list(prj.plots)[i]]
        ax   = plot.axes
        img  = ax.images[0]
        data = np.asarray(img.get_array())
        plot_data[field_name] = data
        if i == 0:
            plot_data_shape = np.shape(data)
                    
    if 'empty_data' not in plot_data.keys():
        plot_data['empty_data'] = np.zeros(plot_data_shape)
        
    plot_data['xlim']  = prj.xlim
    plot_data['ylim']  = prj.ylim
    plot_data['width'] = prj.width
                
    return plot_data



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




def write_disk_model_2(flux_centered, save_dir, d_pc = 140, x_scale = 0, y_scale = 0, z_scale = 0,
                     r = None, inc = None, pa = None, zoom = None, T=None, disk_name=None, ax = None, 
                     output_prefix="diskmodel"):
    '''
    Produces a FITS file from a 2D array of flux densities

    Parameters
    -----------
        flux_centered: 2D numpy array
            Flux densities for x and y pixels with size (800,800)


        sav_dir: string
            Path to directory where FITS file should be saved

        d_pc: float
            distance to source in parsecs (automatically set to 140 pc)

        x_span_AU, y_span_AU : float 
            x and y axis lengths in AU (automatically set to 4000 AU each)

        zoom: float
            zoom of YTProjectionPlot (automatically set to 5)

        T: float
            temperature (automatically set to 20K)

    Returns 
    --------
        A FITS file with the relevant data and headers

    '''

   
    # Imports
    from astropy.io import fits
    import numpy as np
    import os
    from datetime import datetime
    
    # Path to location of header file 
    path = "/workspace/budamgunta/snapshots/"
    
    # FITS that contains headers
    header_file = path+"ppdisk672_GHz_50pc_input.fits"
    
    # Extracting headers
    header = fits.getheader(header_file)

    if z_scale == 0:
        # calculating angular resolution per pixel 
        naxis1 = flux_centered.shape[1] # x-axis
        naxis2 = flux_centered.shape[0] # y-axis
        # angular size per pixel (deg) = (AU per pixel) / (distance in pc) / 3600 arcsec per deg
        cdelt1 = (x_scale / (naxis1 * d_pc)) / 3600   # ra angle (in degrees)
        cdelt2 = (y_scale / (naxis2 * d_pc)) / 3600   # dec angle (in degrees)

    elif x_scale == 0:
        # calculating angular resolution per pixel 
        naxis1 = flux_centered.shape[1] # z-axis
        naxis2 = flux_centered.shape[0] # y-axis
        # angular size per pixel (deg) = (AU per pixel) / (distance in pc) / 3600 arcsec per deg
        cdelt1 = (z_scale / (naxis1 * d_pc)) / 3600   # ra angle (in degrees)
        cdelt2 = (y_scale / (naxis2 * d_pc)) / 3600   # dec angle (in degrees)

    elif y_scale == 0:
        # calculating angular resolution per pixel 
        naxis1 = flux_centered.shape[1] # z-axis
        naxis2 = flux_centered.shape[0] # x-axis
        # angular size per pixel (deg) = (AU per pixel) / (distance in pc) / 3600 arcsec per deg
        cdelt1 = (z_scale / (naxis1 * d_pc)) / 3600   # ra angle (in degrees)
        cdelt2 = (x_scale / (naxis2 * d_pc)) / 3600   # dec angle (in degrees)
       
    now = datetime.now()
    # Changing headers to match our needs
    header['NAXIS']   = 2                                     # 2D axis
    header['NAXIS1']  = naxis1                                # 800 pixels in x direction
    header['NAXIS2']  = naxis2                                # 800 pixels in y direction
    header['BUNIT']   = 'mJy/pixel / Brightness (pixel) unit' # changing units to mJy
    header['CDELT1']  = cdelt1                                # ra angle 
    header['CDELT2']  = cdelt2                                # dec angle
    header['ORIGIN']  = 'CASA version 6.6.1'                   # version of CASA
    header['DATE']    =  now.strftime("%Y-%m-%dT%H:%M:%S")    # date when fits file was generated
    
          

    # Deleting headers
    del header['NAXIS3'] # deleting 3rd axis
    del header['NAXIS4'] # deleting 4th axis

    # file naming 
    name_parts = [output_prefix]
    if disk_name:
        name_parts.append(f"{disk_name}")
    if r is not None:
        name_parts.append(f"r{r}")
    if inc is not None:
        name_parts.append(f"inc{inc}")
    if pa is not None:
        name_parts.append(f"PA{pa}")
    if zoom is not None:
        name_parts.append(f"zoom{zoom}")
    if T is not None:
        name_parts.append(f"T{T}K")
    if ax is not None:
        name_parts.append(f"{ax} axis")

    filename = "_".join(name_parts) + ".fits"

    # Save to a specific directory 
    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)
    # Join path and write the file
    full_path = os.path.join(save_dir, filename)

    fits.writeto(full_path, flux_centered, header, overwrite=True)
    print(f"FITS file written successfully: {full_path}")



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

