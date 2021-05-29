Toolbox contents
==================

An overview of scripts and features contained in the github repository:

* scripts
	* annot2classes.m: helps to organise Freesurfer annotation labels into a parcellation scheme
	* bigbrain_to_fsaverage.sh: called by bigbrainwarp
	* bigbrain_to_icbm.sh: called by bigbrainwarp
	* bigbrainsurf_to_icbm.sh: called by bigbrainwarp
	* compile_profiles.py: collates and saves out intensities into profiles
	* demo_dockerbased.sh: key examples of transformations using the docker installation
	* demo_gitbased.sh: walkthrough of the toolbox utilities using the github installation
	* fsaverage_to_bigbrain.sh: called by bigbrainwarp
	* icbm_to_bigbrain.sh: called by bigbrainwarp
	* icbm_to_bigbrainsurf.sh: called by bigbrainwarp
	* init.sh: initialises the environment
	* io_mesh.py: scripts from `Surface Tools <https://github.com/kwagstyl/surface_tools>`_ that help with loading .obj files
	* nn_surface_indexing.mat: contains mesh decimation output
	* obj2fs.sh: wrapper script to convert .obj surface meshes to a freesurfer style mesh (.pial), which can be loaded into Freeview for visualisation 	
	* sample_intensity_profiles.sh: wrapper script for generating staining intensity profiles
	* txt2curv.sh: wrapper script to convert .txt files to .curv, helpful for visualisation with Freesurfer

* spaces:
	* bigbrain: original histological space, includes surfaces and volumes
	* bigbrainsym: stereotaxic registration of BigBrain to ICBM152 as part of first BigBrain release, includes surfaces and volumes
	* icbm: volumetric data algined to the symmetric ICBM2009b atlas, as well as surfaces from civet
	* fsaverage: surface data on fsaverage
	* fs_LR: surface data on fs_LR 32k

* tutorials
	* gradients: scripts for "Tutorial 1: Comparing BigBrain- and MRI-derived cortical gradients on a common surface"
	* confluence: scripts for "Tutorial 2: Cytoarchitectural characterisation of functional communities"
	* communities: scripts for "Tutorial 3: Variations in resting state functional connectivity along a histological axis"


Preprocessed data can be found across various spaces

.. list-table::
   :widths: 50 50 50
   :header-rows: 1

   * - Data
     - What is it?
     - In which spaces?
   * - profiles.txt
     - cell-staining intensities sampled at each vertex and across 50 equivolumetric surfaces. This is stored as a single vector to reduce the size. Reshape to 50 rows for use. 
     - bigbrain
   * - profiles.txt
     - cell-staining intensities sampled at each vertex and across 50 equivolumetric surfaces. This is stored as a single vector to reduce the size. Reshape to 50 rows for use. 
     - bigbrain
   * - gray*327680*
     - pial surface (Amunts et al. 2013)
     - bigbrain, bigbrainsym
   * - white*327680*
     - white matter boundary (Amunts et al. 2013)
     - bigbrain, bigbrainsym
   * - rh.confluence
     - continuous surface that includes isocortex and allocortex (hippocampus) from `Paquola et al., 2020 <https://elifesciences.org/articles/60673>`_
     - bigbrain
   * - Hist-G*
     - first two eigenvectors of cytoarchtiectural differentitation derived from BigBrain 
     - bigbrain, fsaverage, fs_LR, icbm
   * - Micro-G1
     - first eigenvector of microstructural differentitation derived from quantitative in-vivo T1 imaging
     - bigbrain, fsaverage
   * - Func-G*
     - first threee eigenvectors of functional differentitation derived from rs-fMRI
     - bigbrain, fsaverage
   * - Yeo2011_7Networks_N1000
     - 7 functional clusters from `Yeo & Krienen et al., 2011 <https://doi.org/10.1152/jn.00338.2011>`_
     - bigbrain
   * - Yeo2011_17Networks_N1000
     - 17 functional clusters from `Yeo & Krienen et al., 2011 <https://doi.org/10.1152/jn.00338.2011>`_
     - bigbrain
   * - layer*_thickness
     - Approximate layer thicknesses estimated from `Wagstyl et al., 2020 <https://doi.org/10.1371/journal.pbio.3000678>`_
     - bigbrain, fsaverage, fs_LR
