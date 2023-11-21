"""src/glucose/utils/files.py

This file contains all utils to work with to handle and
process files.

Methods
-------
    list_files(extension, recursive=True)
        list all files in a certain directory

"""

# General imports
import glob

# Typing imports
from typing import List


def list_files(path: str, extension: str = "*", recursive: bool = True) -> List[str]:
    """
    Lists the current files in a directory according to the `extension`,
    and if required recursive.

    Parameters
    ----------
    path: str
        the path which it should lists files in
    extension: str
        which extensions
    recursive: bool
        whether it should do this recursive or not


    Returns
    -------
    List[str]
        returns a list with all full paths to the files

    """
    return [f for f in glob.glob("{}**/*{}".format(path, extension), recursive=recursive)]
