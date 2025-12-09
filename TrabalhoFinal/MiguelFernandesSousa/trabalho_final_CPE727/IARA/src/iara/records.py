"""
IARA Records Module

This module provides functionality to access information about the audio record collections of IARA.
"""
import enum
import os
import typing
import abc

import numpy as np
import pandas as pd

class Rain(enum.Enum):
    """
    Enum representing rain noise with various intensity levels
        HODGES, R. P. Underwater Acoustics Analysis, Design and Performance of SONAR.
        Reino Unido: John Wiley and Sons, Ltd, 2010.
    """
    NO_RAIN = 0
    LIGHT = 1 #(<1 mm/h)
    MODERATE = 2 #(<5 mm/h)
    HEAVY = 3 #(<10 mm/h)
    VERY_HEAVY = 4 #(<100 mm/h)

    def __str__(self) -> str:
        return str(self.name).rsplit(".", maxsplit=1)[-1].capitalize().replace("_", " ")

    @staticmethod
    def classify(values: np.array) -> np.array:
        """Classify a vector of rain intensity data in mm/H.

        Args:
            values (np.array): Vector of rain intensity data in mm/H.

        Returns:
            np.array: Vector of data classified according to the enum.
        """
        return np.select(
            [values == 0, values < 1, values < 5, values < 10],
            [Rain.NO_RAIN, Rain.LIGHT, Rain.MODERATE, Rain.HEAVY],
            Rain.VERY_HEAVY
        )

class SeaState(enum.Enum):
    """
    Enum representing sea state noise with different states.
        HODGES, R. P. Underwater Acoustics Analysis, Design and Performance of SONAR.
        Reino Unido: John Wiley and Sons, Ltd, 2010.
    """
    _0 = 0 # Wind < 0.75 m/s
    _1 = 1 # Wind < 2.5 m/s
    _2 = 2 # Wind < 4.4 m/s
    _3 = 3 # Wind < 6.9 m/s
    _4 = 4 # Wind < 9.8 m/s
    _5 = 5 # Wind < 12.6 m/s
    _6 = 6 # Wind < 19.3 m/s
    _7 = 7 # Wind < 26.5 m/s

    def __str__(self) -> str:
        return str(self.value)

    @staticmethod
    def classify_by_wind(values: np.array) -> np.array:
        """Classify a vector of wind intensity data in m/s.

        Args:
            values (np.array): Vector of wind intensity data in m/s.

        Returns:
            np.array: Vector of data classified according to the enum.
        """
        return np.select(
            [values < 0.75, values < 2.5, values < 4.4, values < 6.9, values < 9.8,
                values < 12.6, values < 19.3],
            [SeaState._0, SeaState._1, SeaState._2, SeaState._3, SeaState._4,
                SeaState._5, SeaState._6],
            SeaState._7
        )

class Collection(enum.Enum):
    """Enum representing the different audio record collections of IARA."""
    A = 0
    OS_NEAR_CPA_IN = 0
    B = 1
    OS_NEAR_CPA_OUT = 1
    C = 2
    OS_FAR_CPA_IN = 2
    D = 3
    OS_FAR_CPA_OUT = 3
    OS_CPA_IN = 4
    OS_CPA_OUT = 5
    OS_SHIP = 6

    E = 7
    OS_BG = 7
    OS = 8

    F = 9
    GLIDER_CPA_IN = 9
    G = 10
    GLIDER_CPA_OUT = 10
    GLIDER_SHIP = 11

    H = 12
    GLIDER_BG = 12
    GLIDER = 13

    COMPLETE = 14

    def __str__(self) -> str:
        """Return a string representation of the collection."""
        return str(self.name).rsplit(".", maxsplit=1)[-1]

    def _get_info_filename(self, only_sample: bool = False) -> str:
        """Return the internal filename for collection information."""
        return os.path.join(os.path.dirname(__file__),
                            "dataset_info",
                            "iara.csv" if not only_sample else "iara_sample.csv")

    def get_selection_str(self) -> str:
        """Get string to filter the 'Dataset' column."""
        if self == Collection.OS_CPA_IN:
            return Collection.A.get_selection_str() + \
                "|" + Collection.C.get_selection_str()

        if self == Collection.OS_CPA_OUT:
            return Collection.B.get_selection_str() + \
                 "|" + Collection.D.get_selection_str()

        if self == Collection.OS_SHIP:
            return Collection.OS_CPA_IN.get_selection_str() + \
                "|" + Collection.OS_CPA_OUT.get_selection_str()

        if self == Collection.OS:
            return Collection.OS_SHIP.get_selection_str()+ \
                "|" + Collection.OS_BG.get_selection_str()

        if self == Collection.GLIDER_SHIP:
            return Collection.F.get_selection_str() + \
                "|" + Collection.G.get_selection_str()

        if self == Collection.GLIDER:
            return Collection.GLIDER_SHIP.get_selection_str() + \
                "|" + Collection.GLIDER_BG.get_selection_str()

        return str(self.name).rsplit(".", maxsplit=1)[-1]

    def get_prettier_str(self) -> str:
        """Get a prettier string representation of the collection."""
        labels = [
            "near cpa in",
            "near cpa out",
            "far cpa in",
            "far cpa out",
            "cpa in",
            "cpa out",
            "os ship",
            "os bg",
            "os",
            "glider cpa in",
            "glider cpa out",
            "glider ship",
        ]
        return labels[self.value]

    def to_df(self, only_sample: bool = False) -> pd.DataFrame:
        """Get information about the collection as a DataFrame.

        Args:
            only_sample (bool, optional): If True, provides information about the sampled
                collection. If False, includes information about the complete collection.
                Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing detailed information about the collection.
        """
        df = pd.read_csv(self._get_info_filename(only_sample=only_sample), na_values=[" - "])
        return df.loc[df['Dataset'].str.contains(self.get_selection_str())]

class Filter():
    """Abstract base class representing a filter to apply on a collection."""
    @abc.abstractmethod
    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the collection based on selected values present in a column.

        Args:
            input_df (pd.DataFrame): The input DataFrame to be filtered.
            
        Returns:
            pd.DataFrame: A filtered DataFrame.
        """

class LabelFilter():
    """Class representing a filter based on values present in a column on a collection."""
    def __init__(self,
                 column: str,
                 values: typing.List[str]):
        """
        Parameters:
        - column (str): Name of the column for selection.
        - values (List[str]): List of values for selection.
        """
        self.column = column
        self.values = values

    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the collection based on selected values present in a column.

        Args:
            input_df (pd.DataFrame): The input DataFrame to be filtered.
            
        Returns:
            pd.DataFrame: A filtered DataFrame containing only the rows with values 
                        present in the specified column.
        """
        return input_df.loc[input_df[self.column].isin(self.values)]
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, LabelFilter):
            return (self.column == other.column and
                    self.values == other.values)
        return False

class Target(Filter):
    DEFAULT_TARGET_HEADER = 'Target'
    def __init__(self,
                 n_targets: int,
                 include_others: bool):
        self.n_targets = n_targets
        self.include_others = include_others

    def get_n_targets(self) -> int:
        """Return the number of targets."""
        return self.n_targets + (1 if self.include_others else 0)

    @abc.abstractmethod
    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        """

    def grouped_column(self) -> str:
        return self.DEFAULT_TARGET_HEADER

class LabelTarget(LabelFilter, Target):
    """Class representing a training target for a dataset."""

    def __init__(self,
                 column: str,
                 values: typing.List[str],
                 include_others: bool = False):
        """
        Parameters:
        - column (str): Name of the column for selection.
        - values (List[str]): List of values for selection.
        - include_others (bool): Indicates whether other values should be compiled as one
            and included or discarded. Default is to discard.
        """
        LabelFilter.__init__(self, column=column, values=values)
        Target.__init__(self, n_targets=len(values), include_others=include_others)

    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the collection based on selected values present in a column.

        Args:
            input_df (pd.DataFrame): The input DataFrame to be filtered.

        Returns:
            pd.DataFrame: A filtered DataFrame containing only the rows with values present in the
                specified column.

        Notes:
            - The target values are assigned based on the unique values present in the 'self.column'
                of the DataFrame.
            - Each unique value is mapped to an integer index, starting from 0.
            - If a value is not found in the 'self.values' list, it is assigned the index equal
                to the length of 'self.values'.
        """
        if not self.include_others:
            input_df = LabelFilter.apply(self, input_df)

        input_df = input_df.assign(**{self.DEFAULT_TARGET_HEADER:
            input_df[self.column].map({value: index for index, value in enumerate(self.values)})})
        input_df[self.DEFAULT_TARGET_HEADER] = \
            input_df[self.DEFAULT_TARGET_HEADER].fillna(len(self.values))
        input_df[self.DEFAULT_TARGET_HEADER] = \
            input_df[self.DEFAULT_TARGET_HEADER].astype(int)
        return input_df

    def grouped_column(self) -> str:
        return self.column

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Target):
            return (self.column == other.column and
                    self.values == other.values and
                    self.include_others == other.include_others)
        return False

class GenericTarget(Target):

    def __init__(self,
                 n_targets : int,
                 function: typing.Callable[[pd.DataFrame],int],
                 include_others: bool = False):
        super().__init__(n_targets=n_targets, include_others=include_others)
        self.function = function

    def apply(self, input_df: pd.DataFrame) -> pd.DataFrame:

        input_df[self.DEFAULT_TARGET_HEADER] = input_df.apply(self.function, axis=1)

        if not self.include_others:
            input_df[self.DEFAULT_TARGET_HEADER] = \
                input_df[self.DEFAULT_TARGET_HEADER].fillna(self.n_targets)
        else:
            input_df = input_df.dropna(subset=[self.DEFAULT_TARGET_HEADER])

        input_df[self.DEFAULT_TARGET_HEADER] = \
            input_df[self.DEFAULT_TARGET_HEADER].astype(int)
        return input_df

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GenericTarget):
            return (self.n_targets == other.n_targets and
                    self.include_others == other.include_others)
        return False

class CustomCollection:
    """Class representing a selection of collection with targets."""

    def __init__(self,
                 collection: Collection,
                 target: Target,
                 filters: typing.Union[typing.List[Filter], Filter] = None,
                 only_sample: bool = False):
        """
        Parameters:
        - collection (Collection): Collection to be used.
        - target (Target): Target selection for training.
        - filters (List[Filter], optional): List of filters to be applied.
            Default is use all collection.
        - only_sample (bool, optional): Use only data available in sample collection.
            Default is False.
        """
        self.collection = collection
        self.target = target
        self.filters = [] if filters is None else \
                (filters if isinstance(filters, list) else [filters])
        self.only_sample = only_sample

    def __str__(self) -> str:
        """Return a string representation of the CustomCollection object."""
        return str(self.to_df())

    def to_df(self) -> pd.DataFrame:
        """
        Generate a DataFrame with information from the collection based on specified filters
            and target configuration.

        Returns:
            pd.DataFrame: DataFrame containing the collection information after applying filters
                and target mapping.
        """
        df = self.collection.to_df(only_sample=self.only_sample)
        for filt in self.filters:
            df = filt.apply(df)

        df = self.target.apply(df)
        return df

    def to_compiled_df(self, df=None) -> pd.DataFrame:

        df = self.to_df() if df is None else df
        if self.target.include_others:
            df_label = df[df['Target'] != self.target.get_n_targets() - 1]
            df_others = df[df['Target'] == self.target.get_n_targets() - 1]

            df_label = df_label.groupby(self.target.grouped_column()).size().reset_index(name='Qty')
            new_row = pd.DataFrame({self.target.grouped_column(): ['Others'], 'Qty': [df_others.shape[0]]})

            df = pd.concat([df_label, new_row])

        else:
            df = df.groupby(self.target.grouped_column()).size().reset_index(name='Qty')

        return df

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CustomCollection):
            return (self.collection == other.collection and
                    self.target == other.target and
                    self.filters == other.filters and
                    self.only_sample == other.only_sample)
        return False