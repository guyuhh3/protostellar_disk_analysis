#!/usr/bin/env python
# coding: utf-8
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

# SNAPSHOT 319
# Data directory
snapdir = r"/data1/Summer_student/snapshot_data_radmc3d/snapshot_files/nonideal_OAH/"

# Snapshot number
i = 319

# Function to get snapshot filename
def get_fname_snap(i, snapdir, verbose=True):
    fname = os.path.join(snapdir, 'snapshot_{0:03d}.hdf5'.format(i))
    if verbose:
        print('filename: {0:s}'.format(fname))
    return fname

# Load snapshot data
snapshot_data = f.load_snapshot_data(i, snapdir, verbose=False)
m_unit = snapshot_data['m_unit']
l_unit = snapshot_data['l_unit']  # already in cm
print(f' the l_unit is {l_unit}')
print(f' the m_unit is {m_unit}')
rho_unit = m_unit / l_unit**3     # g/cm³
rho_cgs = snapshot_data['gas']['rho'] * rho_unit

# Use the particle masses directly
Msun = 1.989e33
gas_mass_cgs = snapshot_data['gas']['m'] * m_unit  # in grams
dust_mass = gas_mass_cgs * 0.01  # in grams
print("Dust mass total = ", np.sum(dust_mass) / m_unit, " Msun")

# Load with yt
yt.set_log_level(50)
unit_base = {
    'UnitMagneticField_in_gauss': 1e4,
    'UnitLength_in_cm': l_unit,
    'UnitMass_in_g': 1.989e33,
    'UnitVelocity_in_cm_per_s': 100.0
}
fname = os.path.join(snapdir, f"snapshot_{i:03d}.hdf5")
ds = yt.load(fname, unit_base=unit_base)

# Run gizmo_carver
gizmo_directory = "/data1/Summer_student/snapshot_data_radmc3d/gizmo_carver-main/"
os.chdir(gizmo_directory)
subprocess.run(["python", "src_319/main_gizmo_carver.py"])

snapdir = r"/data1/Summer_student/snapshot_data_radmc3d/snapshot_files/nonideal_OAH/"
output_dir = f'/data1/Summer_student/snapshot_data_radmc3d/snapshot_files/nonideal_OAH/RADMC_inputs_snapshot_{i}/'
os.makedirs(output_dir, exist_ok=True)

# Grid dust density
# Extract gas particle data with unit conversion
pos = snapshot_data['gas']["X"] * l_unit    # cm
mass = snapshot_data['gas']["m"] * m_unit   # g

# Use yt domain edges for bounding box in code units (convert to cm below)
xmin, ymin, zmin = ds.domain_left_edge.to("cm").value
xmax, ymax, zmax = ds.domain_right_edge.to("cm").value


# Grid resolution
nx, ny, nz = 800, 800, 800

# Convert cell size to cm
dx = (xmax - xmin) / nx  # cm
dy = (ymax - ymin) / ny  # cm
dz = (zmax - zmin) / nz  # cm
cell_volume = dx * dy * dz  # cm^3

# Initialize dust mass grid in Fortran order
dust_to_gas_ratio = 0.01
grid_mass = np.zeros((nx, ny, nz), dtype=np.float64, order='F')

print(f"Position range in cm: x[{pos[:,0].min()}, {pos[:,0].max()}], y[{pos[:,1].min()}, {pos[:,1].max()}], z[{pos[:,2].min()}, {pos[:,2].max()}]")
print(f"Grid domain in cm: x[{xmin}, {xmax}], y[{ymin}, {ymax}], z[{zmin}, {zmax}]")

# Deposit dust mass onto grid
outside_count = 0
for j in range(pos.shape[0]):
    xidx = int((pos[j, 0] - xmin) / dx)
    yidx = int((pos[j, 1] - ymin) / dy)
    zidx = int((pos[j, 2] - zmin) / dz)

    if (0 <= xidx < nx) and (0 <= yidx < ny) and (0 <= zidx < nz):
        grid_mass[xidx, yidx, zidx] += mass[j] * dust_to_gas_ratio
    else:
        outside_count += 1

print(f"Particles outside grid: {outside_count} / {pos.shape[0]}")

# Convert grid mass to g again (if it wasn't in g already)
grid_mass_cgs = grid_mass  # Already in g if mass was in g
# If you used raw mass (code units), uncomment the next line instead:
# grid_mass_cgs = grid_mass * m_unit

# Convert to dust density (g/cm³)
dust_density_cgs = grid_mass_cgs / cell_volume

# Sanity check
print("Dust density stats:")
print("  Min:", np.min(dust_density_cgs), "g/cm**3")
print("  Max:", np.max(dust_density_cgs), "g/cm**3")
print("  Mean:", np.mean(dust_density_cgs), "g/cm**3")

# Flatten array in Fortran order (z fastest)
dust_density_flat = dust_density_cgs.flatten(order='F')

# Output file path
outdir = f"/data1/Summer_student/snapshot_data_radmc3d/snapshot_files/nonideal_OAH/RADMC_inputs_snapshot_{i}"
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, "dust_density.inp")

# Write dust_density.inp
with open(outfile, "w") as f:
    f.write("1\n")                                 # iformat
    f.write(f"{nx*ny*nz}\n")                       # nrcells
    f.write("1\n")                                 # nrspecies
    for val in dust_density_flat:
        f.write(f"{val:.12e}\n")

# Print total mass check
total_dust_mass_msun = np.sum(grid_mass_cgs) / m_unit
print("Dust mass from grid densities (Msun):", total_dust_mass_msun)


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
os.environ["PATH"] += os.pathsep + "/home/chenghanhsieh/localbin/radmc3d-2.0-master/src"
os.chdir(output_dir)
if not os.path.exists(os.path.join(output_dir, "radmc3d.inp")):
    print("Warning: radmc3d.inp not found in output directory!")

# === Generate and Save Images ===
orientations = [
    ["90", "90", "snapshot_319_x_rad_temp.fits"],
    ["90", "0", "snapshot_319_y_rad_temp.fits"],
    ["0", "0", "snapshot_319_z_rad_temp.fits"]
]

for incl, phi, fname in orientations:
    radmc3dPy.image.makeImage(
        npix=800,
        incl=float(incl),
        phi=float(phi),
        posang=0.,
        wav=1266.8872,
        sizeau=800,
        fluxcons=True,
        nostar=False
    )
    img = radmc3dPy.image.readImage()
    img.image = np.squeeze(img.image)
    
   # Apply a floor to avoid blank images
    floor_value = 1e-30
    img.image[img.image < floor_value] = floor_value
    
    
    # Write FITS
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