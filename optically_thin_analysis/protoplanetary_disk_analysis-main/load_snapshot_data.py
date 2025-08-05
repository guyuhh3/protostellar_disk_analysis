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
