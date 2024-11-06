"""A collection of general utility functions for the package."""

import os


def yes_or_no(question: str) -> bool:
    """Utility function to ask a yes/no question from the user.

    Parameters
    ----------
    question : str
        String for user query.

    Returns
    -------
    bool
        Boolean of True for 'yes' and False for 'no'.
    """

    while "the answer is invalid":
        reply = str(input(question + " (y/n): ")).lower().strip()
        if reply[:1] == "y":
            return True
        if reply[:1] == "n":
            return False
    return False


def get_type_of_object(path: str) -> str:
    """Return the type of object at the given path

    Parameters
    ----------
    path : str
        object location

    Returns
    -------
        The typ of the object (file, directory, other, nonexistant)
    """
    if os.path.exists(path):
        if os.path.isdir(path):
            return "directory"
        elif os.path.isfile(path):
            return "file"
        else:
            return "other"
    else:
        return "nonexistant"
