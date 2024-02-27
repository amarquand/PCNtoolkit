def yes_or_no(question):
    """
    Utility function for getting yes/no action from the user.

    :param question: String for user query.

    :return: Boolean of True for 'yes' and False for 'no'.


    """

    while "the answer is invalid":
        reply = str(input(question + " (y/n): ")).lower().strip()
        if reply[:1] == "y":
            return True
        if reply[:1] == "n":
            return False
