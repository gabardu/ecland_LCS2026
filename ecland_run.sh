#!/bin/bash
# Script to run the ecland model on pre-downloaded forcing and initial conditions.

# To compile the model from source, load these modules first (commented out here
# because compilation is a one-time step separate from running):
#module load prgenv/intel intel/2021.4 cmake/3.25 ninja/1.11.1 hpcx-openmpi/2.9 netcdf4/4.9.1 ecbuild/new ecmwf-toolbox/new python3/new

#====== Load modules for intel compiler
# These modules set up the runtime environment needed to execute the model.
# They must match (or be compatible with) the modules used at compile time.
module load prgenv/intel intel/2021.4 python3/3.10.10-01  # Intel compiler environment and Python interpreter
module load hpcx-openmpi/2.9 netcdf4/4.9.1               # MPI library for parallel execution and NetCDF I/O library

#====== Setup (change these as required)
GROUP=LCS2026                          # Experiment group name; used to locate input data sub-folders and name output directories
FORCING_TYPE=era5                      # Type of atmospheric forcing data driving the land-surface model (here: ERA5 reanalysis)
INPUT_DIR=/perm/paga/ecland_input/     # Root directory containing the 'clim/' and 'forcing/' sub-folders for this experiment
#OUTPUT_DIR=/perm/${USER}/${GROUP}_OptParam/   # Alternative output path (commented out; kept for reference)
OUTPUT_DIR=/perm/${USER}/${GROUP}/     # Directory where model output files will be written (created automatically if needed)
WORK_DIR=/scratch/paga/work_${GROUP}/ # Temporary working directory for intermediate files during the run (fast scratch space)
NLOOP=2                                # Number of spin-up loops: the model runs over the full period this many times to initialise soil/snow states

# Namelist file:
NAMELIST_PATH="/home/${USER}/projects/ecland_opensource/ecland_LCS2026/namelists"  # Directory containing the namelist files
NAMELIST_FILE="namelist_ecland_50R1_ctl"  # Filename of the ecland namelist (model configuration file) to use for this run

eclandExe="/home/paga/projects/ecland_opensource/ecland_eclairs-build/bin/ecland-master"  # Full path to the compiled ecland executable

export PATH=${ecland_ROOT:-/home/paga/projects/ecland_opensource/ecland_LCS2026-build}/bin:$PATH  # Add the ecland binary directory to PATH; falls back to a local build if $ecland_ROOT is not set
export MEM_PER_CPU='16G'               # Memory allocation per CPU core (relevant when running in batch mode via a job scheduler)
export LAUNCH='mpirun -np 1'           # Command used to launch the model; '-np 1' means a single MPI process (increase for parallel runs)

#===============================================
START=$(date +%s)   # Record the wall-clock start time in seconds since epoch (used to measure total runtime)

# Run the ecland experiment for the specified site(s).
# Flag reference:
#   -g  Experiment group name; selects the correct input sub-folders under INPUT_DIR
#   -t  Forcing type (e.g. 'era5'); tells the script which forcing format to expect
#   -i  Root input directory containing 'clim/' and 'forcing/' data
#   -o  Output directory where results (NetCDF files) will be saved
#   -l  Number of spin-up loops over the full time period
#   -n  Full path to the namelist configuration file
#   -x  Full path to the ecland executable binary
#   -w  Working directory for temporary files during execution
#   -s  Site identifier and time range (format: <site_id>_<start_year>-<end_year>);
#       omit this flag entirely to run all available sites
ecland-run-experiment \
    -g ${GROUP} \
    -t ${FORCING_TYPE} \
    -i ${INPUT_DIR} \
    -o ${OUTPUT_DIR} \
    -l ${NLOOP} \
    -n "${NAMELIST_PATH}/${NAMELIST_FILE}" \
    -x ${eclandExe} \
    -w ${WORK_DIR} # \
    #-s "CS-001_2017-2021"

END=$(date +%s)                  # Record the wall-clock end time
DIFF=$(( $END - $START ))        # Calculate elapsed time in seconds
echo -e "\n\n\t Running ecland-run-experiment on all sites took $DIFF seconds\n"  # Print the total runtime

