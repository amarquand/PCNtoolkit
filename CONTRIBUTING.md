# Contributing to *PCNtoolkit*

👋🤗Welcome to the *PCNtoolkit* repository.🤗👋

👍We would first like to thank you for taking the time to contribute!👍<br><br>

### Table of contents

Do you know what you want to get out of this guide? Jump straight to the correct section:

1. [Contribution guide](#contribution-guide)
  * [Issues](#issues)
  * [Labels](#labels)
  * [Enhancements (Comment/Fork/Clone)](#enhancements)
2. [Recognition contributors](#recognition-contributors)<br><br>

## Contribution guide <a name="contribution-guide"></a>
With this guide we hope to make it as easy as possible for you to contribute, if you have any queries that are not discussed below, do not hesitate and open an [issue][link_issues]!

Before you start, you'll need to set up a free [GitHub][link_github] account and sign in.
Here are some [instructions][link_signupinstructions].

### Issues <a name="issues"></a>

Every project on GitHub uses [issues][link_issues] slightly differently.

The following outlines how the *PCNtoolkit* developers think about these tools.

* **Issues** are individual pieces of work that need to be completed to move the project forward.
A general guideline: if you find yourself tempted to write a great big issue that
is difficult to describe as one unit of work, please consider splitting it into two or more issues.

    Issues are assigned [labels](#labels) which explain how they relate to the overall project's
    goals and immediate next steps.

### Labels  <a name='labels'></a>

The current list of issue labels are [here][link_labels] and include:

* [![Question](https://img.shields.io/github/labels/amarquand/PCNtoolkit/question)][link_question] *These issues are questions.*

    If you have any doubts about the design, implementation, packages etc. of the toolbox this is a great place to start a question issue and let us know! 

* [![Bug](https://img.shields.io/github/labels/amarquand/PCNtoolkit/bug)][link_bugs] *These issues point to problems in the project.*

    If you find new a bug, please give as much detail as possible in your issue,
    including steps to recreate the error.
    If you experience the same bug as one already listed,
    please add any additional information that you have as a comment.

* [![Enhancement](https://img.shields.io/github/labels/amarquand/PCNtoolkit/enhancement)][link_enhancement] *These issues are asking for new features and improvements to be considered by the project.*

    Please try to make sure that your requested feature is distinct from any others
    that have already been requested or implemented.
    If you find one that's similar but there are subtle differences,
    please reference the other request in your issue.


### Enhancements <a name="enhancements"></a>

Once you have gone through the toolbox and decided a change needs to be made, you're all set up to make a great contribution! We recommend you follow this workflow to make it a smoother process:

1. **Comment on an existing issue or open a new issue referencing your addition.**<br />
  This allows other members of the *PCNtoolkit* development team to confirm that you aren't
  overlapping with work that's currently underway and that everyone is on the same page
  with the goal of the work you're going to carry out.<br />
  [This blog][link_pushpullblog] is a nice explanation of why putting this work in upfront
  is so useful to everyone involved.
  
1. **[Fork][link_fork] the [PCNtoolkit repository][link_PCNtoolkit] to your profile.**<br />
  This is now your own unique copy of *PCNtoolkit*.
  Changes here won't affect anyone else's work, so it's a safe space to explore edits to the code!
  
1. **[Clone][link_clone] your forked PCNtoolkit repository to your machine/computer.**<br />
  While you can edit files [directly on github][link_githubedit], sometimes the changes
  you want to make will be complex and you will want to use a [text editor][link_texteditor]
  that you have installed on your local machine/computer.
  In order to work on the code locally, you must clone your forked repository.<br />  
  To keep up with changes in the PCNtoolkit repository,
  add the ["upstream" PCNtoolkit repository as a remote][link_addremote]
  to your locally cloned repository.
    ```Shell
    git remote add upstream https://github.com/amarquand/PCNtoolkit.git
    ```
    Make sure to [keep your fork up to date][link_updateupstreamwiki] with the upstream repository.<br />  
    For example, to update your developmental branch on your local cloned repository:  
      ```Shell
      git fetch upstream
      git checkout dev
      git merge upstream/dev
      ```

1. **Make and test the changes you've discussed.**<br />
  Make all the changes on the dev (developmental) branch by:
    ```Shell
    git checkout dev
    ```
    Or branch out and create [separate branches][link_branches] for the updates you made. 
    
    Try to keep the changes focused: it is generally easy to review changes that address one feature or bug at a time.
    
    Don't forget to ❗️**test**❗️ all the changes you made by running the [testing scripts][link_tests].
    Once you are satisfied with your local changes, [add/commit/push them][link_add_commit_push]
    to the developmental branch on your forked repository.


1. **Submit a [pull request][link_pullrequest].**<br />
  One of our members will take a look at the changes you proposed and hopefully, after a bit of dialogue, will merge them into the main codebase! <br />

## Recognition contributors <a name="recognition-contributors"></a>

We want to recognize all the work everyone has put in over the years to make the toolbox what it is now. If you are logged in on GitHub you can take a peek at our amazing contributors via the [live contributors page][link_contributorslive].

## Thank you
**Thank you!** for taking the time and we wish you an amazing day/week/year 👋☀️😸.

<br>

<br>

*&mdash; Based on contributing guidelines from the [STEMMRoleModels][link_stemmrolemodels] and [fmriprep][link_fmriprep] models.*


[link_PCNtoolkit]: https://github.com/amarquand/PCNtoolkit
[link_issues]: https://github.com/amarquand/PCNtoolkit/issues
[link_labels]: https://github.com/amarquand/PCNtoolkit/labels
[link_question]: https://github.com/amarquand/PCNtoolkit/labels/question
[link_bugs]: https://github.com/amarquand/PCNtoolkit/labels/bug
[link_enhancement]: https://github.com/amarquand/PCNtoolkit/labels/enhancement
[link_tests]: https://github.com/amarquand/PCNtoolkit/tree/dev/tests

[link_signupinstructions]: https://help.github.com/articles/signing-up-for-a-new-github-account
[link_pushpullblog]: https://www.igvita.com/2011/12/19/dont-push-your-pull-requests/
[link_texteditor]: https://en.wikipedia.org/wiki/Text_editor
[link_contributorslive]: https://labhr.github.io/hatrack/#repo=amarquand/PCNtoolkit

[link_github]: https://github.com/
[link_githubedit]: https://help.github.com/articles/editing-files-in-your-repository
[link_addremote]: https://help.github.com/articles/configuring-a-remote-for-a-fork
[link_updateupstreamwiki]: https://help.github.com/articles/syncing-a-fork/
[link_branches]: https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/
[link_add_commit_push]: https://help.github.com/articles/adding-a-file-to-a-repository-using-the-command-line
[link_pullrequest]: https://help.github.com/articles/creating-a-pull-request-from-a-fork
[link_fork]: https://help.github.com/articles/fork-a-repo/
[link_clone]: https://help.github.com/articles/cloning-a-repository

[link_stemmrolemodels]: https://github.com/KirstieJane/STEMMRoleModels
[link_fmriprep]: https://github.com/nipreps/fmriprep
