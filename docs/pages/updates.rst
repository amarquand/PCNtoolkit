.. _updates:

.. title:: List of updates

Updates
==================

April 7, 2021
------------------------------------------

Transformed staining intensity profiles to fs_LR and fsaverage (our first user request 🎂)

::

    ↪ added spaces/fs_LR/profiles_fs_LR.txt				|  @caseypaquola
    ↪ added spaces/fsaverage/profiles_fsaverage.txt			|  @caseypaquola



March 30, 2021
------------------------------------------
Added ICBM surface, allows transformations between bigbrain surface and icbm volume too. Created a wrapper script for generating staining intensity profiles

::

    ↪ added icbm surface template from civet				|  @caseypaquola
    ↪ added out_type option in bigbrainwarp				|  @caseypaquola
    ↪ created sample_intensity_profiles.sh				|  @caseypaquola



March 24, 2021
------------------------------------------
Transforming data with BigBrainWarp is greatly simplified. Users need only interact with the bigbrainwarp function. 

::

    ↪ created bigbrainwarp base function				|  @caseypaquola
    ↪ expanded volume-based interpolation options			|  @caseypaquola



March 11, 2021
------------------------------------------
Cortical gradients are now available in volume space! We used registration fusion to project the gradient from fsaverage to icbm (`Wu et al., 2018 <https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/registration/Wu2017_RegistrationFusion>`_)

::

    ↪ Hist, Micro and Func gradients added to /spaces/icbm/		|  @caseypaquola



March 4, 2021
------------------------------------------
Estimated layer thicknesses from Wagstyl et al., 2020 surfaces (ftp://bigbrain.loris.ca/BigBrainRelease.2015/Layer_Segmentation/3D_Surfaces/April2019/) and transformed to standard surface templates

::

    ↪ Pre-computed BigBrain layer thicknesses in /spaces/bigbrain/	|  @caseypaquola
    ↪ Transformed layer thicknesses in fsaverage and /fs_LR		|  @caseypaquola


February 25, 2021
------------------------------------------
Changed all surface-based transformations to multi-modal surface matching, using new registrations from Lindsay Lewis. More details `here <https://bigbrainproject.org/docs/4th-bb-workshop/20-06-26-BigBrainWorkshop-Lewis.pdf>`_.

::

    ↪ Overhaul of surface transformation scripts and documentation	|  @caseypaquola



February 8, 2021
------------------------------------------
Expanded the surface transformation functionality. Now supports (i) multi-modal surface registration via a parcel-based transformation and (ii) fs_LR for vertex-based nearest neighbour interpolation.  

::

    ↪ Added approach option for surface based transformations 		|  @caseypaquola
    ↪ Added surface option for fs_LR and fsaverage compatibility    	|  @caseypaquola
    ↪ Updated documention for surface transformations              	|  @caseypaquola
