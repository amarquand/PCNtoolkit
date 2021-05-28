.. include:: links.rst

========================
FAQ, Tips, and Tricks
========================

Should I run quality control of my images before running *fMRIPrep*?
--------------------------------------------------------------------
Yes. You should do so before any processing/analysis takes place.

Oftentimes (more often than we would like), images have fatal artifacts and problems.

Some exclusion criteria for data quality should be pre-specified before QC and any screening
of the original data.
Those exclusion criteria must be designed in agreement with the goals and challenges of the
experimental design.
For instance, when it is planned to run some cortical thickness analysis, images should be excluded
even when they present the most subtle ghosts or other artifacts that may introduce biases in surface
reconstruction.
However, if the same artifactual data was planned to be used just as a reference for spatial
normalization, some of those artifacts should be noted, but may not grant exclusion of the data.

When using publicly available datasets, an additional concern is that images may have gone through
some kind of preprocessing (see next question).


Running subjects in parallel
----------------------------
When running several subjects in parallel, and depending on your settings, fMRIPrep may
fall into race conditions.
A symptomatic output looks like: ::

  FileNotFoundError: [Errno 2] No such file or directory: '/scratch/03201/jbwexler/openneuro_fmriprep/data/ds000003_work/ds000003-download/derivatives/fmriprep-1.4.0/fmriprep/logs/CITATION.md'

If you would like to run *fMRIPrep* in parallel on multiple subjects please use
`this method <https://neurostars.org/t/updated-fmriprep-workaround-for-running-subjects-in-parallel/6677>`__.

How much CPU time and RAM should I allocate for a typical fMRIPrep run?
-----------------------------------------------------------------------
The recommended way to run fMRIPrep is to process one subject per container instance. A typical preprocessing run
without surface processing with FreeSurfer can be completed in about 2 hours with 4 CPUs or in about 1 hour with 16 CPUs.
More than 16 CPUs do not translate into faster processing times for a single subject. About 8GB of memory should be
available for a single subject preprocessing run.

Below are some benchmark data that have been computed on a high performance cluster compute node with Intel E5-2683 v4
CPUs and 64 GB of physical memory:

.. figure:: _static/fmriprep_benchmark.svg

**Compute Time**: time in hours to complete the preprocessing for all subjects. **Physical Memory**: the maximum of RAM usage
used across all fMRIPrep processes as reported by the HCP job manager. **Virtual Memory**: the maximum of virtual memory used
across all fMRIPrep processes as reported by the HCP job manager. **Threads**: the maximum number of threads per process as
specified with ``–omp-nthreads`` in the fMRIPrep command.

The above figure illustrates that processing 2 subjects in 2 fMRIPrep instances with 8 CPUs each is approximately as fast as
processing 2 subjects in one fMRIPrep instance with 16 CPUs. However, on a distributed compute cluster, the two 8 CPU
instances may be allocated faster than the single 16 CPU instance, thus completing faster in practice. If more than one
subject is processed in a single fMRIPrep instance, then limiting the number of threads per process to roughly the
number of CPUs divided by the number of subjects is most efficient.

.. _upgrading:

A new version of *fMRIPrep* has been published, when should I upgrade?
----------------------------------------------------------------------
We follow a philosophy of releasing very often, although the pace is slowing down
with the maturation of the software.
It is very likely that your version gets outdated over the extent of your study.
If that is the case (an ongoing study), then we discourage changing versions.
In other words, **the whole dataset should be processed with the same version (and
same container build if they are being used) of *fMRIPrep*.**

On the other hand, if the project is about to start, then we strongly recommend
using the latest version of the tool.

In any case, if you can find your release listed as *flagged* in `this file
of our repo <https://github.com/nipreps/fmriprep/blob/master/.versions.json>`__,
then please update as soon as possible.

I'm running *fMRIPrep* via Singularity containers - how can I troubleshoot problems?
------------------------------------------------------------------------------------
We have extended `this documentation <singularity.html>`__ to cover some of the most
frequent issues other Singularity users have been faced with.
Generally, users have found it hard to `get TemplateFlow and Singularity to work
together <singularity.html#singularity-tf>`__.

