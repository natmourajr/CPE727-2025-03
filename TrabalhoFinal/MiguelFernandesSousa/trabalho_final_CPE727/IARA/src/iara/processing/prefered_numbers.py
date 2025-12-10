"""
This module provides functions to convert numeric values to engineering notation,
with the ability to specify units and precision.
"""
import math

MAGNITUDE_PREFIX = ["y", "z", "a", "f", "p", "n", "μ", "m", "",
                    "k", "M", "G", "T", "P", "E", "Z", "Y"]
MAGNITUDE_PREFIX_OFFSET = 8

def get_engineering_notation(value: float, unit: str) -> str:
    """
    Convert a numeric value to engineering notation with a specified unit.

    Parameters:
        value (float): The numeric value to be converted.
        unit (str): The unit to be appended to the converted value.

    Returns:
        str: The value in engineering notation with the unit.
    """
    if value == 0:
        return get_engineering_notation_with_precision(value, 1, unit)

    exp = math.log10(abs(value))
    index = math.floor(exp / 3.0)
    base = value / math.pow(10, index * 3)

    return get_engineering_notation_with_precision(value,
                                                   2 - int(math.floor(math.log10(abs(base)))), unit)

def get_engineering_notation_with_precision(value: float, precision: int, unit: str) -> str:
    """
    Convert a numeric value to engineering notation with a specified precision and unit.

    Parameters:
        value (float): The numeric value to be converted.
        precision (int): The number of decimal places in the converted value.
        unit (str): The unit to be appended to the converted value.

    Returns:
        str: The value in engineering notation with the specified precision and unit.
    """
    if value == 0:
        if unit != "":
            return "0 " + unit
        return "0"

    exp = math.log10(abs(value))
    index = math.floor(exp / 3.0)
    base = value / math.pow(10, index * 3)

    string = ""

    if precision == 0:
        string = "{:3.0f}"
    elif precision == 1:
        string = "{:3.1f}"
    elif precision == 2:
        string = "{:3.2f}"
    elif precision == 3:
        string = "{:3.3f}"
    elif precision == 4:
        string = "{:3.4f}"
    elif precision == 5:
        string = "{:3.5f}"
    elif precision == 6:
        string = "{:3.6f}"
    elif precision == 7:
        string = "{:3.7f}"
    elif precision == 8:
        string = "{:3.8f}"
    elif precision == 9:
        string = "{:3.9f}"
    else:
        index = len(MAGNITUDE_PREFIX)
    index += MAGNITUDE_PREFIX_OFFSET

    if index < 0 or index >= len(MAGNITUDE_PREFIX):
        string = "{:.05e}"
        index = MAGNITUDE_PREFIX_OFFSET
        base = value

    ret_str = string.format(base)

    if unit == "" and MAGNITUDE_PREFIX[index] == "":
        return ret_str

    return ret_str + " " + MAGNITUDE_PREFIX[index] + unit
