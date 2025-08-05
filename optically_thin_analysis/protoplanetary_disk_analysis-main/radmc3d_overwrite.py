    
    
    
def dust_density(snapshot_data, i, model = "nonideal_OA"):
    # imports
    import numpy as np
    import os
    
    # loading data (N number of gas particles)
    pos = snapshot_data['gas']['X']     # shape (N, 3)
    
    m_unit   = snapshot_data['m_unit']
    l_unit   = snapshot_data['l_unit']
    rho_unit = m_unit / l_unit**3
    rho      = snapshot_data['gas']['rho'] * rho_unit   # shape (N,  )
    
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    x_size = x.shape[0]
    y_size = y.shape[0]
    z_size = z.shape[0]
    
    
    # dust-to-gas mass ratio
    mass_ratio = 0.01
    dust_densities = rho * mass_ratio
    
    # grid resolution
    nx, ny, nz = 512, 512, 512
    
    # compute grid bounds (i.e. axis bounds) 
    xmin, ymin, zmin = pos.min(axis=0)
    xmax, ymax, zmax = pos.max(axis=0)
    
    # create grid edges each box has two sides 
    x_edges = np.linspace(xmin, xmax, nx + 1)
    y_edges = np.linspace(ymin, ymax, ny + 1)
    z_edges = np.linspace(zmin, zmax, nz + 1)
    
    # initialize grid (grid zeros to populate)
    density_grid = np.zeros((nx, ny, nz))
    
    # assign particles to grid cells
    ix = np.searchsorted(x_edges, pos[:, 0], side='right') - 1
    iy = np.searchsorted(y_edges, pos[:, 1], side='right') - 1
    iz = np.searchsorted(z_edges, pos[:, 2], side='right') - 1
     
    # clip indices to stay within bounds
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    iz = np.clip(iz, 0, nz - 1)
    
    # Accumulate dust densities (writes densities in to proper grid pixels)
    for j in range(len(dust_densities)):
        density_grid[ix[j], iy[j], iz[j]] += dust_densities[i]
    
    # Normalize by cell volume 
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz
    cell_volume = dx * dy * dz
    density_grid /= cell_volume

    # Write to RADMC-3D format
    base_dir = '/workspace/budamgunta/snapshots/all_snapshots/output_files/'
    work_dir = f'{model}/RADMC_inputs_snapshot_{i}'
    full_path = os.path.join(base_dir, work_dir)
    os.makedirs(full_path, exist_ok=True)

    
    with open(os.path.join(full_path, 'dust_density.inp'), 'w') as f:
        f.write('1\n')  # Format number
        f.write(f'{nx * ny * nz}\n')  # Total number of grid cells
        f.write('1\n')  # Number of dust species
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    f.write(f'{density_grid[ix, iy, iz]:13.6e}\n')
    
    print(f"'dust_density.inp' has been written to: {full_path}")













def stars_inp(snapshot_data, i, model = "nonideal_OA"):
    # writing stars.inp file
    import numpy as np
    # output path
    base_dir = '/workspace/budamgunta/snapshots/all_snapshots/output_files/'
    work_dir = f'{model}/RADMC_inputs_snapshot_{i}'
    full_path = os.path.join(base_dir, work_dir)
    os.makedirs(full_path, exist_ok=True)
    os.chdir(full_path)
    
    # Confirm you're in the right place
    print("Current directory:", os.getcwd())
    
    # Assuming you have sink particle data:
    sink_ids = snapshot_data['stars']['ids']
    sink_mass = snapshot_data['stars']['m']
    sink_pos_pc = snapshot_data['stars']['X'] 
    
    # Physical constants
    c = 2.998e14          # speed of light [micron/s]
    h = 6.626e-27         # Planck constant [erg s]
    kB = 1.38e-16         # Boltzmann constant [erg/K]
    rsun_cm = 6.96e10     # solar radius [cm]
    pc_to_cm = 3.086e18   # parsec to cm
    
    # Read wavelengths
    with open("wavelength_micron.inp", "r") as f:
        n_lambda = int(f.readline().strip())
        wavelengths = np.array([float(f.readline().strip()) for _ in range(n_lambda)])  # microns
    
    frequencies = c / wavelengths  # Hz
    
    # Estimate star properties

    # Estimate radius and temperature using simplified power-law model
    rstar_cm = 3.0 * (sink_mass ** 0.8) * rsun_cm  # radius in cm
    tstar = 4000 * (sink_mass ** 0.1)              # temperature in K
    
    # Blackbody spectrum
    def planck_nu(nu, T):
        B_nu = (2 * h * nu**3) / c**2 / (np.exp(h * nu / (kB * T)) - 1)
        return B_nu  # erg/s/cm^2/Hz/sr
    
    nstars = len(sink_ids)
    source_spectra = np.zeros((nstars, n_lambda))
    
    for j in range(nstars):
        B_nu = planck_nu(frequencies, tstar[j]) * np.pi  # Integrate over solid angle
        area = 4 * np.pi * rstar_cm[j]**2
        source_spectra[j, :] = B_nu * area
    
    # Write stars.inp in iformat = 2
    with open("stars.inp", "w") as f:
        f.write("2\n")
        f.write(f"{nstars} {n_lambda}\n")
        for i in range(nstars):
            x_cm = sink_pos_pc[i, 0] * pc_to_cm
            y_cm = sink_pos_pc[i, 1] * pc_to_cm
            z_cm = sink_pos_pc[i, 2] * pc_to_cm
            f.write(f"{rstar_cm[i]:.6e} {sink_mass[i]:.6e} {x_cm:.6e} {y_cm:.6e} {z_cm:.6e}\n")
        for wl in wavelengths:
            f.write(f"{wl:.8e}\n")
        for i in range(nstars):
            for val in source_spectra[i]:
                f.write(f"{val:.6e}\n")
    
    print(f"'stars.inp' has been written to: {full_path}")


    



def modify_gas_temp(input_file, output_file, new_temp):
    """
    Reads a RADMC-3D gas_temperature.inp file, replaces all temperature values
    with a uniform value, and writes the result to a new file.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Parse header
    format_number = lines[0].strip()
    num_cells = int(lines[1].strip())

    # Replace all temperature values with the new uniform value
    new_values = [f"{new_temp:.9e}\n"] * num_cells

    # Write to output file
    with open(output_file, 'w') as f:
        f.write(f"{format_number}\n")
        f.write(f"{num_cells}\n")
        f.writelines(new_values)

    print(f"Modified gas_temperature.inp written to: {output_file}")




def modify_dust_temp(input_file, output_file, new_temp):
    """
    Reads a RADMC-3D dust_temperature.inp file, replaces all temperature values
    with a uniform value, and writes the result to a new file.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Parse header
    format_number = lines[0].strip()
    num_cells = int(lines[1].strip())

    # Replace all temperature values with the new uniform value
    new_values = [f"{new_temp:.9e}\n"] * num_cells

    # Write to output file
    with open(output_file, 'w') as f:
        f.write(f"{format_number}\n")
        f.write(f"{num_cells}\n")
        f.writelines(new_values)

    print(f"Modified dust_temperature.inp written to: {output_file}")


