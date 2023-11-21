"""src/glucose/utils/processing.py

This scripts contains all functionality to read, process
and format data in a state to be used in the pipeline.
"""

# Pandas imports
import pandas as pd

# Datetime imports
from datetime import date, timedelta, datetime


def activpal_time_to_date(matlab_float: float) -> datetime:
    """
    Convert an ActivPal (MATLAB) time to a datetime format in
    Python.

    Parameters
    ----------
    matlab_float: float
        the raw matlab value to transform

    Returns
    -------
    datetime
        returns the transformed value
    """

    matlab_dt: datetime = datetime.fromordinal(int(matlab_float)) + \
        timedelta(days=matlab_float % 1) - \
        timedelta(days=366)

    # 09/02: we observed that all ActivPal times were
    # one day off: so for now we manually correct this
    # but we should examine this more properly.
    matlab_dt = matlab_dt - timedelta(days=1)

    return matlab_dt.replace(microsecond=0)


def add_years(d: datetime, years: int) -> datetime:
    """
    Returns a date that's years after the date. Returns the same
    calendar date (e.g. 11 February 2000, 11 February 2008) if it exists.
    Otherwise it will use the following day (e.g. February 29 to March 1).

    Parameters
    ----------
    d: datetime.datetime
        the original datetime.datetime object to transform
    years: int
        the amount of years to add

    Returns
    -------
    datetime
        returns the transformed object

    """

    try:
        return d.replace(year=d.year + years)
    except ValueError as e:
        print('help', e)
        # noinspection PyTypeChecker
        return d + (date(d.year + years, 1, 1)) - date(d.year, 1, 1)


def nearest_item(items: pd.Series, pivot: object) -> object:
    """
    Returns the nearest item of any `pd.Series` which is closest
    to the pivot object.

    Parameters
    ----------
    items: pd.Series
        the pd.Series containing the items
    pivot
        the item which will be looked up

    Returns
    -------
    object
        returns the object in the pd.Series containing the closest value

    """
    return min(items, key=lambda x: abs(x - pivot))
