Frequently Asked Questions
====================================

**What's the difference between BigBrain and BigBrainSym?** The original volumetric reconstruction of BigBrain is tilted compared to a typical brain housed in a skull (see below). This has to do with tissue deformations of the specimen post-mortem.

To facilitate comparisons with standard neuroimaging spaces, BigBrain was nonlinearly transformed to ICBM152 space, resulting in BigBrainSym. When parsing arguments to scripts in this toolbox, we use the "bigbrain" for BigBrain and "bigbrainsym" for BigBrainSym.

.. figure:: ./images/FAQ_BigBrainSym.png
   :height: 150px
   :align: center


**What space is X in?**  For data provided on BigBrainWarp, the space corresponds the directory location. For example, anything in spaces/bigbrain/ is in native bigbrain space. If the file contains a suffix (eg: "_bigbrain"), that means the data was transformed from a different space to that space. 

**How can I build equivolumetric surfaces?** This can be performed using the `python-based surface tools <https://github.com/kwagstyl/surface_tools/tree/v1.0.0>`_, which we've also translated for `matlab-based pipelines <https://github.com/MICA-MNI/micaopen/blob/master/cortical_confluence/scripts/equivolumetric_surfaces.m>`_. 

**How can I obtain staining intensity profiles?** We've pre-generated a standard set of staining intensity profiles for BigBrain using 50 surfaces between the pial and white matter, the 100um resolution volume of BigBrain and conservative smoothing. If you would like to make your own, for example, using different surfaces or volume, try the `sample_intensity_profiles.sh <https://github.com/MICA-MNI/micaopen/blob/master/BigBrainWarp/scripts/sample_intensity_profiles.sh>`_. It requires an volume to sample as well as upper and lower surfaces (in .obj format) as input. Then specify the number of surfaces, and the function will pull on surface_tools (see above) to generate equivolumetric surfaces, sample intensities and compile these as a profiles. Example use: bash sample_intensity_profiles.sh -in_vol full8_100um_optbal.mnc -upper_surf pial_left_327680.obj -lower_surf white_left_327680.obj -wd tests -num_surf 50

**Are there regions of BigBrain that should be treated with caution?** There is a small tear in the left entorhinal cortex of BigBrain, which affects the pial surface construction as well as microstructure profiles in that region. For region of interest studies, it is always a good idea to carry out a detailed visual inspection. Try out the `EBRAINS interactive viewer <https://interactive-viewer.apps.hbp.eu/?templateSelected=Big+Brain+%28Histology%29&parcellationSelected=Cytoarchitectonic+Maps+-+v2.4&cNavigation=0.0.0.-W000..2_ZG29.-ASCS.2-8jM2._aAY3..BSR0..PDY1%7E.rzeq%7E.5qQV..15ye>`_

**What causes "Error: opening MINC file BigBrainHist-to-ICBM2009sym-nonlin_grid_0.mnc" when the file exists?** This can be caused by using minc1 instead of minc2. Check which mincresample version is running from your terminal using "which mincresample". Ensure this come from the minc2 installation and not from another location on your computer. For instance, freesurfer contains a mincresample that is minc1 that will throw this error. 




