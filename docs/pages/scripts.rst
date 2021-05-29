How to warp
===============

Integrating histological information with in vivo neuroimaging can deepen our understanding of neuroanatomy and structure-function coupling in the human brain. 

Running BigBrainWarp
********************************

First, `install the package <https://bigbrainwarp.readthedocs.io/en/latest/pages/installation.html>`_
To run BigBrainWarp using docker, you must mount a working directory and a directory with a personal Freesurfer license. You can obtain a Freesurfer license at `https://surfer.nmr.mgh.harvard.edu/registration.html <https://surfer.nmr.mgh.harvard.edu/registration.html>`_.

.. code-block:: bash

    # with docker (change the local locations of the mounts accordingly)
    docker run -it --rm -v /local/directory/with/freesurfer_license:/license \
        -v /local/directory/with/data/:/BigBrainWarp/tests \
        caseypaquola/bigbrainwarp bigbrainwarp

    # without docker
    bigbrainwarp


The following arguments can be used with BigBrainWarp

.. list-table::
   :widths: 25 50 50 50
   :header-rows: 1

   * - Parameter
     - Description	
     - Conditions	
     - Options
   * - in_space	
     - Space of input data	
     - Required	
     - bigbrain, bigbrainsym, icbm, fsaverage, fs_LR 
   * - out_space	
     - Space of output data	
     - Required	
     - bigbrain, bigbrainsym, icbm, fsaverage, fs_LR 
   * - wd
     - Path to working directory
     - Required	
     - 
   * - in_vol	
     - Full path to input data, whole brain volume.	
     - Requires either in_vol, or in_lh and in_rh	
     - Permitted formats: mnc, nii or nii.gz
   * - ih_lh	
     - Full path to input data, left hemisphere surface.
     - Requires either in_vol, or in_lh and in_rh	
     - Permitted formats: label.gii, annot, shape.gii, curv or txt
   * - ih_rh	
     - Full path to input data, left hemisphere surface.
     - Requires either in_vol, or in_lh and in_rh	
     - Permitted formats: label.gii, annot, shape.gii, curv or txt
   * - interp	
     - Interpolation method
     - Required for in_vol.
     - Optional for txt input. Not permitted for other surface inputs.	For in_vol, can be trilinear (default), tricubic, nearest or sinc. For txt, can be linear or nearest
   * - out_name	
     - Prefix for output files	
     - Required for surface input. 
     - Optional for volume input, otherwise defaults to prefix of input file
   * - out_type	
     - Specifies whether output in surface or volume space 	
     - Optional function for bigbrain and bigbrainsym output. Otherwise, defaults to the same type as the input.  	
     - surface, volume


The BigBrainWarp function currently wraps the following range of transformations

.. image:: ./images/bbw_workflow.png
   :height: 300px
   :align: center


Example transformations in volume space
********************************

.. code-block:: bash

	# for example, transformation of a bigbrain to icbm can take the form
	bigbrainwarp --in_space bigbrain --out_space icbm --wd /project/ --in data.nii --interp trilinear

	# in contrast, transformation from icbm to bigbrainsym could be
	bigbrainwarp --in_space icbm --out_space bigbrainsym --wd /project/ --in data.mnc --interp sinc


BigBrainWarp utilises a recently published nonlinear transformation Xiao et al., (2019)
If you use volume-based transformations in BigBrainWarp, please cite:
Xiao, Y., et al. 'An accurate registration of the BigBrain dataset with the MNI PD25 and ICBM152 atlases'. Sci Data 6, 210 (2019). https://doi.org/10.1038/s41597-019-0217-0


Example transformations for surface-based data
***************************************

Surface-based transformation can be enacted using multi-modal surface matching; a spherical registration approach. Ongoing work by Lewis et al., involves optimisation of registration surafces between BigBrain and standard surface templates. These are available at `ftp://bigbrain.loris.ca/BigBrainRelease.2015/BigBrainWarp_Support <ftp://bigbrain.loris.ca/BigBrainRelease.2015/BigBrainWarp_Support>`_. More details on procedure can be also found on the following `poster <https://drive.google.com/file/d/1vAqLRV8Ue7rf3gsNHMixFqlLxBjxtmc8/view?usp=sharing>`_ and `slides <https://drive.google.com/file/d/11dRgtttd2_FdpB31kDC9mUP4WCmdcbbg/view?usp=sharing>`_.
The functions currently support fsaverage and fs_LR as standard imaging templates for input or output.

.. code-block:: bash

	# for example, transformation of a bigbrain to fsaverage can take the form
	bigbrainwarp --in_space bigbrain --out_space fsaverage --wd /project/ --in_lh lh.data.label.gii --in_rh rh.data.label.gii --out_name data

	# in contrast, transformation from icbm to bigbrainsym could be
	bigbrainwarp --in_space fs_LR --out_space bigbrain --wd /project/ --in_lh lh.data.label.txt --in_rh rh.data.label.txt --out_name data --interp linear


If you use surface-based transformations in BigBrainWarp, please cite:
Lewis, L.B., et al. 'A multimodal surface matching (MSM) surface registration pipeline to bridge atlases across the MNI and the Freesurfer/Human Connectome Project Worlds' OHBM, Virtual (2020)






