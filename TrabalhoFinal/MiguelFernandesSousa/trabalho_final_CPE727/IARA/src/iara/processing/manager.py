"""Module for handling access to processed data from a audio dataset.

This module defines a class, `AudioFileProcessor`, which facilitates access to processed data by
providing methods for loading and retrieving dataframes based on specified parameters. It supports
the processing and normalization of audio data, allowing users to work with either window-based or
image-based input types.
"""
import os
import enum
import typing
import tqdm
import json
import hashlib

import PIL
import pandas as pd
import numpy as np
import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt
import matplotlib.colors as color
import tikzplotlib as tikz

import iara.utils as iara_utils
import iara.processing.analysis as iara_proc
import iara.processing.prefered_numbers as iara_pn


def get_iara_id(file:str) -> int:
    """
    Default function to extracts the ID from the given file name.

    Parameters:
        file (str): The file name without extension.

    Returns:
        int: The extracted ID.
    """
    return int(file.rsplit('-',maxsplit=1)[-1])


class PlotType(enum.Enum):
    """Enum defining plot types."""
    SHOW_FIGURE = 0
    EXPORT_RAW = 1
    EXPORT_PLOT = 2
    EXPORT_TEX = 3

    def __str__(self):
        return str(self.name).rsplit('_', maxsplit=1)[-1].lower()

class AudioFileProcessor():
    """ Class for handling acess to process data from a dataset. """

    def __init__(self,
                data_base_dir: str,
                data_processed_base_dir: str,
                normalization: iara_proc.Normalization,
                analysis: iara_proc.SpectralAnalysis,
                n_pts: int = 1024,
                n_overlap: int = 0,
                n_mels: int = 256,
                decimation_rate: int = 1,
                extract_id: typing.Callable[[str], int] = get_iara_id,
                frequency_limit: float = None,
                integration_overlap=0,
                integration_interval=None
                ) -> None:
        """
        Parameters:
            data_base_dir (str): Base directory for raw data.
            data_processed_base_dir (str): Base directory for process data.
            normalization (iara_proc.Normalization): Normalization object.
            analysis (iara_proc.Analysis): Analysis object.
            n_pts (int): Number of points for use in analysis.
            n_overlap (int): Number of points to overlap for use in analysis.
            n_mels (int): Number of Mel frequency bins for use in analysis.
            decimation_rate (int): Decimation rate for use in analysis when mel based.
            extract_id (Callable[[str], str]): Function to extract ID from a file name without
                extension. Default is split based on '-' em get last part of the name
            frequency_limit (float): The frequency limit to be considered in the data
                processing result. Default is fs/2
        """
        self.data_base_dir = data_base_dir
        self.data_processed_base_dir = data_processed_base_dir
        self.normalization = normalization
        self.analysis = analysis
        self.n_pts = n_pts
        self.n_overlap = n_overlap
        self.n_mels = n_mels
        self.decimation_rate = decimation_rate
        self.extract_id = extract_id
        self.frequency_limit = frequency_limit
        self.integration_overlap = integration_overlap
        self.integration_interval = integration_interval

        self._check_dir()

    def _get_output_dir(self) -> str:
        return os.path.join(self.data_processed_base_dir,
                                  str(self.analysis) + "_" + self._get_hash())

    def _get_hash(self) -> str:
        converted = json.dumps(self._to_dict(), sort_keys=True)
        hash_obj = hashlib.md5(converted.encode())
        return hash_obj.hexdigest()

    def _to_dict(self) -> typing.Dict:
        return {
            'data_base_dir': self.data_base_dir,
            'data_processed_base_dir': self.data_processed_base_dir,
            'normalization': str(self.normalization),
            'analysis': str(self.analysis),
            'n_pts': self.n_pts,
            'n_overlap': self.n_overlap,
            'n_mels': self.n_mels,
            'decimation_rate': self.decimation_rate,
            'frequency_limit': self.frequency_limit,
            'integration_overlap': self.integration_overlap,
            'integration_interval': self.integration_interval,
        }

    def _save(self, path: str = None):
        config_file = os.path.join(self._get_output_dir() if path is None else path, "config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self._to_dict(), f, indent=4)

    def _check_dir(self) -> None:
        if not os.path.exists(self._get_output_dir()):
            os.makedirs(self._get_output_dir(), exist_ok=True)
            self._save()

    def _find_raw_file(self, file_id: int) -> str:
        """
        Finds the raw file associated with the given ID.

        Parameters:
            file_id (int): The ID to search for.

        Returns:
            str: The path to the raw file.

        Raises:
            UnboundLocalError: If the file is not found.
        """
        for root, _, files in os.walk(self.data_base_dir):
            for file in files:
                filename, extension = os.path.splitext(file)
                if extension == ".wav" and self.extract_id(filename) == file_id:
                    return os.path.join(root, file)
        raise UnboundLocalError(f'file {file_id} not found in {self.data_base_dir}')

    def _process(self, file_id: int) -> typing.Tuple[np.array, np.array, np.array]:

        file = self._find_raw_file(file_id = file_id)

        fs, data = scipy_wav.read(file)

        if data.ndim != 1:
            data = data[:,0]

        power, freqs, times = self.analysis.apply(data = data,
                                                  fs = fs,
                                                  n_pts = self.n_pts,
                                                  n_overlap = self.n_overlap,
                                                  n_mels = self.n_mels,
                                                  decimation_rate = self.decimation_rate,
                                                  integration_overlap = self.integration_overlap,
                                                  integration_interval = self.integration_interval)

        power = self.normalization(power)

        if self.frequency_limit:
            index_limit = next((i for i, freq in enumerate(freqs)
                                if freq > self.frequency_limit), len(freqs))
            freqs = freqs[:index_limit]
            power = power[:index_limit,:]

        return power, freqs, times

    def get_data(self, file_id: int) -> typing.Tuple[pd.DataFrame, np.array]:

        os.makedirs(self._get_output_dir(), exist_ok=True)
        filename = os.path.join(self._get_output_dir(), f'{file_id}.pkl')

        if os.path.exists(filename):
            data = pd.read_pickle(filename)
            return data['df'], data['times']

        power, freqs, times = self._process(file_id)

        columns = [f'f {i}' for i in range(len(freqs))]
        df = pd.DataFrame(power.T, columns=columns)
        df.to_pickle(filename)

        data_to_save = {'df': df, 'times': times}
        pd.to_pickle(data_to_save, filename)

        return df, times

    def get_complete_df(self,
               file_ids: typing.Iterable[int],
               targets: typing.Iterable) -> typing.Tuple[pd.DataFrame, pd.Series]:
        """
        Retrieve data for the given file IDs.

        Parameters:
            - file_ids (Iterable[int]): The list of IDs to fetch data for;
                a pd.Series of ints can be passed as well.
            - targets (Iterable): List of target values corresponding to the file IDs.
                Should have the same number of elements as file_ids.

        Returns:
            Tuple[pd.DataFrame, pd.Series]:
                - pd.DataFrame: The DataFrame containing the processed data.
                - pd.Series: The Series containing the target values,
                    with the same type as the target input.
        """
        result_df = pd.DataFrame()
        result_target = pd.Series()

        for local_id, target in tqdm.tqdm(
                                list(zip(file_ids, targets)), desc='Get data', leave=False, ncols=120):
            data_df, _ = self.get_data(local_id)
            result_df = pd.concat([result_df, data_df], ignore_index=True)

            replicated_targets = pd.Series([target] * len(data_df), name='Target')
            result_target = pd.concat([result_target, replicated_targets], ignore_index=True)

        return result_df, result_target

    def plot(self,
             file_id: typing.Union[int, typing.Iterable[int]],
             plot_type: PlotType = PlotType.EXPORT_PLOT,
             frequency_in_x_axis: bool=False,
             colormap: color.Colormap = plt.get_cmap('jet'),
             override: bool = False) -> None:
        """
        Display or save images with processed data.

        Parameters:
            file_id (Union[int, Iterable[int]]): ID or list of IDs of the file to plot.
            plot_type (PlotType): Type of plot to generate (default: PlotType.EXPORT_PLOT).
            frequency_in_x_axis (bool): If True, plot frequency values on the x-axis.
                Default: False.
            colormap (Colormap): Colormap to use for the plot. Default: 'jet'.
            override (bool): If True, override any existing saved plots. Default: False.

        Returns:
            None
        """
        if plot_type != PlotType.SHOW_FIGURE:
            output_dir = os.path.join(self._get_output_dir(), str(plot_type))
            os.makedirs(output_dir, exist_ok=True)
            self._save()

        if not isinstance(file_id, int):
            for local_id in tqdm.tqdm(file_id, desc='Plot', leave=False, ncols=120):
                self.plot(
                    file_id = local_id,
                    plot_type = plot_type,
                    frequency_in_x_axis = frequency_in_x_axis,
                    colormap = colormap,
                    override = override)
            return

        if plot_type == PlotType.EXPORT_RAW or plot_type == PlotType.EXPORT_PLOT:
            filename = os.path.join(output_dir,f'{file_id}.png')
        elif plot_type == PlotType.EXPORT_TEX:
            filename = os.path.join(output_dir,f'{file_id}.tex')
        else:
            filename = " "

        if os.path.exists(filename) and not override:
            return

        power, freqs, times = self._process(file_id)

        if frequency_in_x_axis:
            power = power.T

        if plot_type == PlotType.EXPORT_RAW:
            power = colormap(power)
            power_color = (power * 255).astype(np.uint8)
            image = PIL.Image.fromarray(power_color)
            image.save(filename)
            return

        times[0] = 0
        freqs[0] = 0

        n_ticks = 5
        time_labels = [iara_pn.get_engineering_notation(times[i], "s")
                    for i in np.linspace(0, len(times)-1, num=n_ticks, dtype=int)]

        frequency_labels = [iara_pn.get_engineering_notation(freqs[i], "Hz")
                    for i in np.linspace(0, len(freqs)-1, num=n_ticks, dtype=int)]

        time_ticks = [(x/4 * (len(times)-1)) for x in range(n_ticks)]
        frequency_ticks = [(y/4 * (len(freqs)-1)) for y in range(n_ticks)]

        plt.figure()
        plt.imshow(power, aspect='auto', origin='lower', cmap=colormap)
        plt.colorbar()

        if frequency_in_x_axis:
            plt.ylabel('Time')
            plt.xlabel('Frequency')
            plt.yticks(time_ticks)
            plt.gca().set_yticklabels(time_labels)
            plt.xticks(frequency_ticks)
            plt.gca().set_xticklabels(frequency_labels)
            plt.gca().invert_yaxis()
        else:
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.xticks(time_ticks)
            plt.gca().set_xticklabels(time_labels)
            plt.yticks(frequency_ticks)
            plt.gca().set_yticklabels(frequency_labels)

        plt.tight_layout()

        if plot_type == PlotType.SHOW_FIGURE:
            plt.show()
        elif plot_type == PlotType.EXPORT_PLOT:
            plt.savefig(filename)
            plt.close()
        elif plot_type == PlotType.EXPORT_TEX:
            tikz.save(filename)
            plt.close()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AudioFileProcessor):
            return self._get_hash() == other._get_hash()
        return False
