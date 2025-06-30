pcntoolkit.util.paths
=====================

.. py:module:: pcntoolkit.util.paths

.. autoapi-nested-parse::

   Path-related utilities for PCNtoolkit.



Functions
---------

.. autoapisummary::

   pcntoolkit.util.paths.ensure_dir_exists
   pcntoolkit.util.paths.get_default_home_dir
   pcntoolkit.util.paths.get_default_log_dir
   pcntoolkit.util.paths.get_default_save_dir
   pcntoolkit.util.paths.get_default_temp_dir
   pcntoolkit.util.paths.get_save_subdirs


Module Contents
---------------

.. py:function:: ensure_dir_exists(path: str) -> None

   Ensure that a directory exists, creating it if necessary.

   :param path: The directory path to ensure exists
   :type path: :py:class:`str`


.. py:function:: get_default_home_dir() -> str

   Get the default home directory for PCNtoolkit.

   :returns: The default home directory path
   :rtype: :py:class:`str`


.. py:function:: get_default_log_dir() -> str

   Get the default log directory for PCNtoolkit.

   :returns: The default log directory path
   :rtype: :py:class:`str`


.. py:function:: get_default_save_dir() -> str

   Get the default save directory for normative models.

   The save directory is determined in the following order:
   1. PCN_SAVE_DIR environment variable if set
   2. ~/.pcntoolkit/saves if the directory exists or can be created
   3. ./saves as a fallback

   :returns: The default save directory path
   :rtype: :py:class:`str`


.. py:function:: get_default_temp_dir() -> str

   Get the default temp directory for PCNtoolkit.

   The temp directory is determined in the following order:
   1. PCN_TEMP_DIR environment variable if set
   2. ~/.pcntoolkit/temp if the directory exists or can be created
   3. ./temp as a fallback

   :returns: The default temp directory path
   :rtype: :py:class:`str`


.. py:function:: get_save_subdirs(save_dir: str) -> tuple[str, str, str]

   Get the standard subdirectories for saving model data.

   :param save_dir: The base save directory
   :type save_dir: :py:class:`str`

   :returns: Tuple of (model_dir, results_dir, plots_dir)
   :rtype: :py:class:`tuple[str`, :py:class:`str`, :py:class:`str]`


