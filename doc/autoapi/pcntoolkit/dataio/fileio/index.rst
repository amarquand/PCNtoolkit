pcntoolkit.dataio.fileio
========================

.. py:module:: pcntoolkit.dataio.fileio


Attributes
----------

.. autoapisummary::

   pcntoolkit.dataio.fileio.CIFTI_MAPPINGS
   pcntoolkit.dataio.fileio.CIFTI_VOL_ATLAS
   pcntoolkit.dataio.fileio.path
   pcntoolkit.dataio.fileio.path


Functions
---------

.. autoapisummary::

   pcntoolkit.dataio.fileio.create_incremental_backup
   pcntoolkit.dataio.fileio.create_mask
   pcntoolkit.dataio.fileio.file_extension
   pcntoolkit.dataio.fileio.file_stem
   pcntoolkit.dataio.fileio.file_type
   pcntoolkit.dataio.fileio.load
   pcntoolkit.dataio.fileio.load_ascii
   pcntoolkit.dataio.fileio.load_cifti
   pcntoolkit.dataio.fileio.load_nifti
   pcntoolkit.dataio.fileio.load_pd
   pcntoolkit.dataio.fileio.save
   pcntoolkit.dataio.fileio.save_ascii
   pcntoolkit.dataio.fileio.save_cifti
   pcntoolkit.dataio.fileio.save_nifti
   pcntoolkit.dataio.fileio.save_pd
   pcntoolkit.dataio.fileio.vol2vec


Module Contents
---------------

.. py:function:: create_incremental_backup(filepath)

   Create an incremental backup of a file using the `.bak{n}` naming scheme.

   If the file does not exist, an empty file is created at the specified path.
   A backup is then created in the same directory with the following format:
       original_name.bak{n}.original_extension
   where {n} is incremented based on existing backups.


.. py:function:: create_mask(data_array, mask, verbose=False)

   Create a mask from a data array or a nifti file

   Basic usage::

           create_mask(data_array, mask, verbose)

   :param data_array: numpy array containing the data to write out
   :param mask: nifti image containing a mask for the image
   :param verbose: verbose output


.. py:function:: file_extension(filename)

   Determine the file extension of a file (e.g. .nii.gz)

   Basic usage::

                       file_extension(filename)

   :param filename: name of the file to check


.. py:function:: file_stem(filename)

   Determine the file stem of a file (e.g. /path/to/file.nii.gz -> file)

   Basic usage::

                               file_stem(filename)

   :param filename: name of the file to check


.. py:function:: file_type(filename)

   Determine the file type of a file

   Basic usage::

                   file_type(filename)

   :param filename: name of the file to check
   :returns: str: The file type ('cifti', 'nifti', 'text', or 'binary')
   :raises: ValueError if file type is unknown


.. py:function:: load(filename, mask=None, text=False, vol=True)

   Load array from a file.

   A beautiful waterfall of errors.

   Basic usage::

                   load(filename, mask, text, vol)

   :param filename: name of the file to load
   :param mask: nifti image containing a mask for the image
   :param text: whether to write out a text file
   :param vol: whether to load the image as a volume


.. py:function:: load_ascii(filename)

   Load an ascii file into a numpy array

   Basic usage::

           load_ascii(filename)

   :param filename: name of the file to load


.. py:function:: load_cifti(filename, vol=False, mask=None, rmtmp=True)

   Load a cifti file into a numpy array

   Basic usage::

                       load_cifti(filename, vol, mask, rmtmp)

   :param filename: name of the file to load
   :param vol: whether to load the image as a volume
   :param mask: nifti image containing a mask for the image
   :param rmtmp: whether to remove temporary files


.. py:function:: load_nifti(datafile, mask=None, vol=False)

   Load a nifti file into a numpy array

   Basic usage::

                   load_nifti(datafile, mask, vol, verbose)

   :param datafile: name of the file to load
   :param mask: nifti image containing a mask for the image
   :param vol: whether to load the image as a volume
   :param verbose: verbose output


.. py:function:: load_pd(filename)

   Load a csv file into a pandas dataframe

   Basic usage::

                   load_pd(filename)

   :param filename: name of the file to load


.. py:function:: save(data, filename, example=None, mask=None, text=False, dtype=None)

   Save a numpy array to a file

   Basic usage::

               save(data, filename, example, mask, text, dtype)

   :param data: numpy array containing the data to write out
   :param filename: where to store it
   :param example: example file to copy the geometry from
   :param mask: nifti image containing a mask for the image
   :param text: whether to write out a text file
   :param dtype: data type for the output image (if different from the image)


.. py:function:: save_ascii(data, filename)

   Save a numpy array to an ascii file

   Basic usage::

       save_ascii(data, filename)

   :param data: numpy array containing the data to write out
   :param filename: where to store it


.. py:function:: save_cifti(data, filename, example, mask=None, vol=True, volatlas=None)

   Save a cifti file from a numpy array

   Basic usage::

                           save_cifti(data, filename, example, mask, vol, volatlas)

   :param data: numpy array containing the data to write out
   :param filename: where to store it
   :param example: example file to copy the geometry from
   :param mask: nifti image containing a mask for the image
   :param vol: whether to load the image as a volume
   :param volatlas: atlas to use for the volume


.. py:function:: save_nifti(data, filename, examplenii, mask, dtype=None)

   Write output to nifti

   Basic usage::

       save_nifti(data, filename mask, dtype)

   :param data: numpy array containing the data to write out
   :param filename: where to store it
   :param examplenii: nifti to copy the geometry and data type from
   :mask: nifti image containing a mask for the image
   :param dtype: data type for the output image (if different from the image)


.. py:function:: save_pd(data, filename)

   Save a pandas dataframe to a csv file

   Basic usage::

       save_pd(data, filename)

   :param data: pandas dataframe containing the data to write out
   :param filename: where to store it


.. py:function:: vol2vec(dat, mask, verbose=False)

   Vectorise a 3d image

   Basic usage::

               vol2vec(dat, mask, verbose)

   :param dat: numpy array containing the data to write out
   :param mask: nifti image containing a mask for the image
   :param verbose: verbose output


.. py:data:: CIFTI_MAPPINGS
   :value: ('dconn', 'dtseries', 'pconn', 'ptseries', 'dscalar', 'dlabel', 'pscalar', 'pdconn', 'dpconn',...


.. py:data:: CIFTI_VOL_ATLAS
   :value: 'Atlas_ROIs.2.nii.gz'


.. py:data:: path
   :value: b'.'


.. py:data:: path

