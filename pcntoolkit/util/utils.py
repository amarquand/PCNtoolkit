"""A collection of general utility functions for the package."""


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
