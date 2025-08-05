# EDIT: directories, snapshot number, nx, ny, nz, inputs_carver
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os
import subprocess
import functions as f
import yt
import unyt
from scipy.stats import binned_statistic_dd
import radmc3dPy
# Data directory containing snapshot files
snapdir = r"/data1/Summer_student/snapshot_data_radmc3d/snapshot_files/nonideal_OAH_jets"

i = 339

# Load snapshot data
snapshot_data = f.load_snapshot_data(i, snapdir, verbose=False)
m_unit = snapshot_data['m_unit']
l_unit = snapshot_data['l_unit']  # already in cm
rho_unit = m_unit / l_unit**3     # g/cm³
rho_cgs = snapshot_data['gas']['rho'] * rho_unit
dust_rho = rho_cgs * 0.01          # 1% dust-to-gas ratio

# Grid dust mass into fixed-size box
pos = snapshot_data['gas']['X']  # already in cm
mass_per_particle = snapshot_data['gas']['m'] * m_unit  # in grams
dust_mass_per_particle = mass_per_particle * 0.01  # 1% dust-to-gas ratio

# Use the same box size and units as RADMC-3D input
box_size_au = 800  # AU
au_to_cm = 1.495978707e13  # 1 AU in cm
box_size_cm = box_size_au * au_to_cm
half_box = box_size_cm / 2.0

# Center the box on the mean particle position
box_center = np.mean(pos, axis=0)
box_min = box_center - half_box
box_max = box_center + half_box

# Grid resolution
nx, ny, nz = 800, 800, 800
edges = [
    np.linspace(box_min[0], box_max[0], nx + 1),
    np.linspace(box_min[1], box_max[1], ny + 1),
    np.linspace(box_min[2], box_max[2], nz + 1)
]

# Bin dust mass into grid
grid_vals, _, _ = binned_statistic_dd(pos, values=dust_mass_per_particle, statistic='sum', bins=edges)

# Convert to density
dx = (box_max[0] - box_min[0]) / nx
dy = (box_max[1] - box_min[1]) / ny
dz = (box_max[2] - box_min[2]) / nz
cell_volume = dx * dy * dz
grid_vals /= cell_volume  # now in g/cm^3

# Flatten for RADMC-3D
flat_vals = grid_vals.flatten(order='F')

# Write dust_density.inp
output_dir = f'/data1/Summer_student/snapshot_data_radmc3d/snapshot_files/nonideal_OA/RADMC_inputs_snapshot_{i}/'
os.makedirs(output_dir, exist_ok=True)
filepath = os.path.join(output_dir, 'dust_density.inp')
with open(filepath, 'w') as f_out:
    f_out.write("1\n")
    f_out.write(f"{flat_vals.size}\n")
    f_out.write("1\n")
    np.savetxt(f_out, flat_vals, fmt="%13.6e")
print(f"'dust_density.inp' written to: {filepath}")



# Write stars.inp
os.chdir(output_dir)
sink_ids = snapshot_data['stars']['ids']
sink_mass = snapshot_data['stars']['m']
sink_pos_pc = snapshot_data['stars']['X']
rsun_cm = 6.96e10
pc_to_cm = 3.086e18
with open("wavelength_micron.inp", "r") as f_wl:
    n_lambda = int(f_wl.readline().strip())
    wavelengths = np.array([float(f_wl.readline().strip()) for _ in range(n_lambda)])
frequencies = 2.998e14 / wavelengths
rstar_cm = (sink_mass ** 0.8) * rsun_cm
tstar = 4000 + 500 * (sink_mass - 1)
nstars = len(sink_ids)
source_spectra = np.zeros((nstars, n_lambda))
def planck_nu(nu, T):
    h = 6.626e-27
    c = 2.998e14
    kB = 1.38e-16
    return (2 * h * nu**3) / c**2 / (np.exp(h * nu / (kB * T)) - 1)
for i in range(nstars):
    B_nu = planck_nu(frequencies, tstar[i]) * np.pi
    area = 4 * np.pi * rstar_cm[i]**2
    source_spectra[i, :] = B_nu * area
with open("stars.inp", "w") as f_star:
    f_star.write("2\n")
    f_star.write(f"{nstars} {n_lambda}\n")
    for i in range(nstars):
        x_cm = sink_pos_pc[i, 0] * pc_to_cm
        y_cm = sink_pos_pc[i, 1] * pc_to_cm
        z_cm = sink_pos_pc[i, 2] * pc_to_cm
        f_star.write(f"{rstar_cm[i]:.6e} {sink_mass[i]:.6e} {x_cm:.6e} {y_cm:.6e} {z_cm:.6e}\n")
    for wl in wavelengths:
        f_star.write(f"{wl:.8e}\n")
    for i in range(nstars):
        for val in source_spectra[i]:
            f_star.write(f"{val:.6e}\n")
print(f"'stars.inp' has been written to: {output_dir}")



# Run radmc3d for 3 orientation
# Add radmc3d path to environment
os.environ["PATH"] += os.pathsep + "/home/chenghanhsieh/localbin/radmc3d-2.0-master/src"
output_dir = '/data1/Summer_student/snapshot_data_radmc3d/snapshot_files/nonideal_OA/RADMC_inputs_snapshot_343/'

os.chdir(output_dir)
if not os.path.exists(os.path.join(output_dir, "radmc3d.inp")):
    print("Warning: radmc3d.inp not found in output directory!")
    
# === Generate and Save Images ===
import radmc3dPy

orientations = [
    ["90", "90", "snapshot_343_x_rad_temp.fits"],
    ["90", "0", "snapshot_343_y_rad_temp.fits"],
    ["0", "0", "snapshot_343_z_rad_temp.fits"]
]

for incl, phi, fname in orientations:
    # Generate the image using radmc3dPy
    radmc3dPy.image.makeImage(
        npix=1024,
        incl=float(incl),
        phi=float(phi),
        posang=0.,
        wav=1266.8872,
        sizeau=150,
        fluxcons=True,
        nostar=False
    )

    # Read and process the image
    img = radmc3dPy.image.readImage()
    img.image = np.squeeze(img.image)

    # Apply a floor to avoid blank images
    floor_value = 1e-30
    img.image[img.image < floor_value] = floor_value

    # Write to FITS
    img.writeFits(
        fname=fname,
        dpc=140.0,
        coord='00h00m00s +00d00m00s',
        bandwidthmhz=2000.0,
        casa=True,
        nu0=0.0,
        stokes='I',
        fitsheadkeys={'OBJECT': 'DiskModel'},
        ifreq=None
    )

    print(f"{fname} has been written to {output_dir}")

