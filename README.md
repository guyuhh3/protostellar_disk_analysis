# protostellar_disk_substructures
Two methods of analyzing spiral substructures in protostellar from STARFORGE snapshot data.

## File Contents

### optically_thin_analysis
* *protoplanetary_disk_analysis-main*
Contains files to load, extract, and analyze snapshot data along with examples. Many files are borrowed from nina-af's repository, refer to the following link for more details. https://github.com/nina-af/protostellar_disk_analysis/tree/main
    * **analyze_snapshot.py**
    Analysis of cloud snapshot properties.
    * **analye_snapshot_TURBSPHERE.py**
    Analysis of turbulence properties in TURBSPHERE snapshots.
    * **flux_density_calculations.py**
    Functions to compute dust denisty,flux denisty, and write to fits.
    * **load_snapshot_data.py**
    Functions to load snapshots and get data from snapshots or plots.
    * **plot_density_profile.py** 
    Functions to plot denisty profiles. 
    * **plot_density_projection.py**
    Functions to plot denisty projections.
    * **plot_yt_data.py**
    Creates slice plots from yt data.
    * **ppdisk672_GHz_50pc_input.fits**
    File used to extract FITS headers.
    * **protostellar_disk_snapshot_info.txt**
    Text file containing snapshot details.
    * **snapshot_180.hdf5**
Example snapshot file.
    
    
    
* *ppd_analysis_initial.ipynb*
Jupyter Notebook with guide to load snapshot data, compute flux denisty (optically thin model), and write fits files. 



### radmc-3d_analysis
* *executable_scripts*
    * **319_radmc3d.py**
Example script used to write input files and run radmc-3d at three different inclinations including rewriting dust_density.inp, writing stars.inp, and wriitng fits files.
    * **radmc3d_fits.py**
    Example script for running radmc3d and writing fits files if you already have input files. 
    * **radmc3d_overwrite.py**
    Functions to write/overwrite dust_density.inp, stars.inp, gas_temperature.inp, and dust_temperature.inp.


* *gizmo_carver-main*
    * Copied from soffner's repository, access details at the following link.  https://github.com/soffner/gizmo_carver

* *inputs+radmc-3d_tutorial.ipynb*
   * Jupyter notebook tutorial with code on how to write input files for radmc-3d and then run radmc-3d.
