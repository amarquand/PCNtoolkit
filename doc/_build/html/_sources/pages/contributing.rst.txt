Contributing
============

First off, thanks for considering contributing to PCNtoolkit! 🎉👍

The following is a set of guidelines for contributing to PCNtoolkit. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

How Can I Contribute?
---------------------

Reporting Bugs
^^^^^^^^^^^^^^

This section guides you through submitting a bug report. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

Before Submitting A Bug Report
""""""""""""""""""""""""""""""

* Ensure the bug is not already reported by searching on GitHub under `Issues <https://github.com/amarquand/PCNtoolkit/issues>`_

How Do I Submit A Good Bug Report?
""""""""""""""""""""""""""""""""""

Bugs are tracked as `GitHub issues <https://github.com/amarquand/PCNtoolkit/issues>`_. Create an issue and provide the following information:

* Use a clear and descriptive title for the issue to identify the problem
* Describe the exact steps which reproduce the problem in as much detail as possible
* Provide specific examples to demonstrate the steps. Include links to files or GitHub projects, or copy/pasteable snippets

Suggesting Enhancements
^^^^^^^^^^^^^^^^^^^^^^^

This section guides you through submitting an enhancement suggestion, including completely new features and minor improvements to existing functionality.

How Do I Submit A Good Enhancement Suggestion?
""""""""""""""""""""""""""""""""""""""""""""""

Enhancement suggestions are tracked as `GitHub issues <https://github.com/amarquand/PCNtoolkit/issues>`_. Create an issue and provide the following information:

* Use a clear and descriptive title for the issue to identify the suggestion
* Provide a step-by-step description of the suggested enhancement in as much detail as possible
* Provide specific examples to demonstrate the steps

Development Setup
-----------------

1. Fork the repo
2. Clone your fork:

   .. code-block:: bash

       git clone https://github.com/your-username/pcntoolkit.git

3. Install development dependencies:

   .. code-block:: bash

       pip install -e ".[dev]"

Styleguides
-----------

Git Commit Messages
^^^^^^^^^^^^^^^^^^^

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

Python Styleguide
^^^^^^^^^^^^^^^^^

All Python code must adhere to `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_, and we use ``autopep8`` for automatic code formatting. Please see the `autopep8 documentation <https://github.com/hhatto/autopep8>`_ for more details. The autopep8 settings can be found in setup.cfg.

Running Tests
-------------

.. code-block:: bash

    pytest tests/

Additional Notes
----------------

Feel free to propose changes to this document in a pull request.