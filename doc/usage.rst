.. include:: links.rst

.. _Usage :

Usage Notes
===========

Execution 
-----------------------------
The *PCNtoolkit* workflow takes as principal input the path of the design matrix. 

The exact command to run *PCNtoolkit* depends on the Installation_ method.

Example: ::

    PCNtoolkit data/bids_root/ out/ participant -w work/


Command-Line Arguments
----------------------
.. argparse::
   :ref: PCNtoolkit.cli.parser._build_parser
   :prog: PCNtoolkit
   :nodefault:
   :nodefaultconst:


The command-line interface of the docker wrapper
------------------------------------------------

.. argparse::
   :ref: PCNtoolkit_docker.get_parser
   :prog: PCNtoolkit-docker
   :nodefault:
   :nodefaultconst:

Troubleshooting
---------------
Logs and crashfiles are outputted into the
``<output dir>/PCNtoolkit/sub-<participant_label>/log`` directory.

**Support and communication**.
The documentation of this project is found here: http://PCNtoolkit.readthedocs.org/en/latest/.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/andremarquand/PCNtoolkit/issues.

If you have a problem or would like to ask a question about how to use *PCNtoolkit*,
please submit a question to `NeuroStars.org <http://neurostars.org/tags/PCNtoolkit>`_ with an ``PCNtoolkit`` tag.
NeuroStars.org is a platform similar to StackOverflow but dedicated to neuroinformatics.

Previous questions about *PCNtoolkit* are available here:
http://neurostars.org/tags/PCNtoolkit/

To participate in the *PCNtoolkit* development-related discussions please use the
following Gitter page: https://gitter.im/PCNtoolkit/community#
