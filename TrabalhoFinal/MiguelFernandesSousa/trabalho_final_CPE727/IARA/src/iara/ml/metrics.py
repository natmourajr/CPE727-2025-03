"""
Module for model evaluation metrics and result compilation.

This module provides classes for computing evaluation metrics commonly used in model analysis,
as well as for compiling and formatting evaluation results from cross-validation and grid search.
"""
import enum
import typing
import math
import pickle

import numpy as np
import pandas as pd
import dill

import sklearn.metrics as sk_metrics
import scipy.stats as scipy
import matplotlib.pyplot as plt
import matplotlib as mpl

class Metric(enum.Enum):
    """Enumeration representing metrics for model analysis.

    This enumeration provides a set of metrics commonly used for evaluating multiclasses models and
        a eval method to perform this evaluation.
    """
    ACCURACY = 1
    BALANCED_ACCURACY = 2
    MICRO_F1 = 3
    MACRO_F1 = 4
    DETECTION_PROBABILITY = 5
    SP_INDEX = 6
    MACRO_RECALL = 7
    MICRO_RECALL = 8
    MACRO_PRECISION = 9
    MICRO_PRECISION = 10

    def __str__(self):
        """Return the string representation of the Metric enum."""
        return str(self.name).rsplit('.', maxsplit=1)[-1].lower()

    def as_label(self):
        """Return the human-readable label of the metric."""
        en_labels = {
            __class__.ACCURACY: "ACC",
            __class__.BALANCED_ACCURACY: "ACC",
            __class__.MICRO_F1: "F1-score",
            __class__.MACRO_F1: "F1-score",
            __class__.DETECTION_PROBABILITY: "DETECTION_PROBABILITY",
            __class__.SP_INDEX: "SP",
            __class__.MACRO_RECALL: "Recall",
            __class__.MICRO_RECALL: "Micro Recall",
            __class__.MACRO_PRECISION: "Precision",
            __class__.MICRO_PRECISION: "Micro Precision",
        }
        return en_labels[self]

    def compute(self, target: typing.Iterable[int], prediction: typing.Iterable[int]) -> float:
        """Compute the metric value based on the target and prediction.

        Args:
            target (Iterable[int]): True labels.
            prediction (Iterable[int]): Predicted labels.

        Returns:
            float: Computed metric value.
        """
        if self == Metric.ACCURACY:
            return sk_metrics.accuracy_score(target, prediction) * 100

        if self == Metric.BALANCED_ACCURACY:
            return sk_metrics.balanced_accuracy_score(target, prediction) * 100

        if self == Metric.MICRO_F1:
            return sk_metrics.f1_score(target, prediction, average='micro') * 100

        if self == Metric.MACRO_F1:
            return sk_metrics.f1_score(target, prediction, average='macro') * 100

        if self == Metric.MACRO_RECALL:
            return sk_metrics.recall_score(target, prediction, average='macro') * 100

        if self == Metric.MICRO_RECALL:
            return sk_metrics.recall_score(target, prediction, average='micro') * 100

        if self == Metric.MACRO_PRECISION:
            return sk_metrics.precision_score(target, prediction, average='macro') * 100

        if self == Metric.MICRO_PRECISION:
            return sk_metrics.precision_score(target, prediction, average='micro') * 100

        if self == Metric.DETECTION_PROBABILITY:
            cm = sk_metrics.confusion_matrix(target, prediction, labels=list(set(list(target))))
            detection_probabilities = cm.diagonal() / cm.sum(axis=1)
            return np.mean(detection_probabilities) * 100

        if self == Metric.SP_INDEX:
            cm = sk_metrics.confusion_matrix(target, prediction, labels=list(set(list(target))))
            detection_probabilities = cm.diagonal() / cm.sum(axis=1)
            geometric_mean = scipy.gmean(detection_probabilities)
            return np.sqrt(np.mean(detection_probabilities * geometric_mean)) * 100



        raise NotImplementedError(f"Evaluation for Metric {self} is not implemented.")

    @staticmethod
    def compute_all(metric_list: typing.List['Metric'],
                    target: typing.Iterable[int],
                    prediction: typing.Iterable[int]) -> typing.Dict['Metric', float]:
        """Compute all metrics in the given list.

        Args:
            metric_list (List[Metric]): List of metrics to compute.
            target (Iterable[int]): True labels.
            prediction (Iterable[int]): Predicted labels.

        Returns:
            Dict[Metric, float]: Dictionary containing computed metric values for each metric
                in the list.
        """
        dict_values = {}
        for metric in metric_list:
            dict_values[metric] = metric.compute(target, prediction)
        return dict_values

class Test(enum.Enum):
    F_TEST_5x2 = 0
    STD_OVERLAY = 1
    WILCOXON = 2

    @staticmethod
    def std_overlay(sample1: np.ndarray, sample2: np.ndarray, confidence_level: float) -> bool:
        mean1 = np.mean(sample1)
        std1 = np.std(sample1)
        mean2 = np.mean(sample2)
        std2 = np.std(sample2)
        return np.abs(mean1 - mean2) > (std1 + std2)

    @staticmethod
    def f_test_5x2(sample1: np.ndarray, sample2: np.ndarray, confidence_level: float) -> bool:
        #http://rasbt.github.io/mlxtend/user_guide/evaluate/combined_ftest_5x2cv/
        #https://www.cmpe.boun.edu.tr/~ethem/files/papers/NC110804.PDF
        if len(sample1) != 10 or len(sample2) != 10:
            print('sample1: ', len(sample1))
            print('sample2: ', len(sample2))
            raise UnboundLocalError('For Ftest_5x2 must be calculated 10 values')

        p_1_a = sample1[0::2]
        p_2_a = sample1[1::2]

        p_1_b = sample2[0::2]
        p_2_b = sample2[1::2]

        p_1 = p_1_a - p_1_b
        p_2 = p_2_a - p_2_b

        p_mean = (p_1 + p_2)/2
        s_2 = (p_1 - p_mean)**2 + (p_2 - p_mean)**2

        N = (np.sum(p_1**2) + np.sum(p_2**2))

        if N == 0:
            return False, 0

        f = N / (2*np.sum(s_2))

        confidence = scipy.f.cdf(f, 10, 5)

        return f > scipy.f.ppf(confidence_level, 10, 5), confidence

    @staticmethod
    def wilcoxon(sample1: np.ndarray, sample2: np.ndarray, confidence_level: float) -> bool:
        if len(sample1) != len(sample2):
            print('sample1: ', len(sample1))
            print('sample2: ', len(sample2))
            raise UnboundLocalError('Wilcoxon Signed-rank must be calculated with sample with same length')

        if np.sum(sample1 - sample2) == 0:
            return False, 0

        _, p_value = scipy.wilcoxon(sample1, sample2)
        return (1 - p_value) > confidence_level, (1 - p_value)

    def eval(self, sample1: typing.Iterable, sample2: typing.Iterable, confidence_level = 0.95) -> bool: #return true se diferente
        return getattr(self.__class__, self.name.lower())(sample1, sample2, confidence_level)

class CrossValidationCompiler():
    """Class for compiling cross-validation results.

    This class compiles the results of cross-validation evaluations, including metric scores
    for each fold and each metric, and provides methods for formatting the results.
    """

    def __init__(self) -> None:
        """Initialize the CrossValidationCompiler object."""

        self._score_dict = {
            'i_fold':[],
            'n_samples':[],
            'metrics': [],
            'abs_cm': {},
            'rel_cm': {}
        }

        for metric in Metric:
            self._score_dict[str(metric)] = []

    def add(self,
            i_fold: int,
            metric_list: typing.List[Metric],
            target: typing.Iterable[int],
            prediction: typing.Iterable[int]) -> None:
        """Add evaluation results for a fold.

        Args:
            i_fold (int): Fold index.
            metric_list (List[Metric]): List of metrics to compute.
            target (Iterable[int]): True labels.
            prediction (Iterable[int]): Predicted labels.
        """

        self._score_dict['i_fold'].append(i_fold)
        self._score_dict['n_samples'].append(len(target))

        for metric, score in Metric.compute_all(metric_list, target, prediction).items():
            self._score_dict[str(metric)].append(score)

        self._score_dict['metrics'].extend(metric_list)
        self._score_dict['metrics'] = list(set(self._score_dict['metrics']))

        self._score_dict['abs_cm'][i_fold] = sk_metrics.confusion_matrix(target, prediction)
        self._score_dict['rel_cm'][i_fold] = sk_metrics.confusion_matrix(target, prediction, normalize='true')

    def get(self, metric: Metric):
        return self._score_dict[str(metric)]

    def print_abs_cm(self):
        result = ""
        result_matrix = None
        for i_fold in self._score_dict['i_fold']:
            if result_matrix is None:
                result_matrix = self._score_dict['abs_cm'][i_fold].flatten()
            else:
                result_matrix = np.column_stack((result_matrix, self._score_dict['abs_cm'][i_fold].flatten()))

        if len(self._score_dict['i_fold']) != 1:
            mean = np.mean(result_matrix, axis=1)
            std = np.std(result_matrix, axis=1)
        else:
            mean = result_matrix
            std = np.zeros(mean.shape)
        n_elements = self._score_dict['abs_cm'][self._score_dict['i_fold'][0]].shape

        for i in range(n_elements[0]):
            for j in range(n_elements[1]):
                result += f'{mean[n_elements[1]*i + j]:.1f} +- {std[n_elements[1]*i + j]:.1f} \t'
            result += '\n'

        return result
    
    def print_rel_cm(self):
        result = ""
        result_matrix = None
        for i_fold in self._score_dict['i_fold']:
            if result_matrix is None:
                result_matrix = self._score_dict['rel_cm'][i_fold].flatten()
            else:
                result_matrix = np.column_stack((result_matrix, self._score_dict['rel_cm'][i_fold].flatten()))

        if len(self._score_dict['i_fold']) != 1:
            mean = np.mean(result_matrix, axis=1)
            std = np.std(result_matrix, axis=1)
        else:
            mean = result_matrix
            std = np.zeros(mean.shape)
        n_elements = self._score_dict['rel_cm'][self._score_dict['i_fold'][0]].shape

        for i in range(n_elements[0]):
            if i != 0:
                result += '\n'

            for j in range(n_elements[1]):
                result += f'{mean[n_elements[1]*i + j]*100:.2f} +- {std[n_elements[1]*i + j]*100:.2f} \t'

        return result

    def print_cm(self, filename: str = None, relative=True):
        dict_id = 'rel_cm' if relative else 'abs_cm'

        num_folds = len(self._score_dict[dict_id])
        first_matrix = next(iter(self._score_dict[dict_id].values()))
        n_classes = first_matrix.shape[0]

        confusion_matrices_3d = np.zeros((num_folds, n_classes, n_classes))

        for i_fold, rel_cm in self._score_dict[dict_id].items():
            confusion_matrices_3d[i_fold, :, :] = rel_cm

        mean_cm = np.mean(confusion_matrices_3d, axis=0) * (100 if relative else 1)
        std_cm = np.std(confusion_matrices_3d, axis=0) * (100 if relative else 1)


        string = ''
        for i in range(n_classes):
            string = f'{string}\t    {i}:\t'
        print(string)
        for i in range(n_classes):
            string = ''
            for j in range(n_classes):
                string = f'{string}\t{mean_cm[i,j]:.2f} ± {std_cm[i,j]:.2f}'
            print(f'{i}:  {string}')

        if filename is None:
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(mean_cm, cmap=mpl.cm.Blues)
        fig.colorbar(cax)

        for (i, j), val in np.ndenumerate(mean_cm):
            std_val = std_cm[i, j]
            ax.text(j, i, f'{val:.2f} ± {std_val:.2f}', ha='center', va='center', color='black')

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix with Mean ± Std')

        plt.savefig(filename, dpi=300, bbox_inches='tight')

    @staticmethod
    def str_format(values, n_samples=60, tex_format=False) -> str:
        """Format the values as a string.

        Args:
            values: Values to format.
            n_samples (int, optional): Number of samples to compute the decimal places.
                Defaults to 60.
            tex_format (bool, optional): Whether to format as LaTeX. Defaults to False.

        Returns:
            str: Formatted string.
        """
        decimal_places = 2#int(math.log10(math.sqrt(n_samples))+1)
        if tex_format:
            return f'${np.mean(values):.{decimal_places}f} \\pm {np.std(values):.{decimal_places}f}$'

        return f'{np.mean(values):.{decimal_places}f} \u00B1 {np.std(values):.{decimal_places}f}'

    @staticmethod
    def table_to_str(table: typing.List[typing.List[str]]) -> str:
        """Convert the table to a formatted string.

        Args:
            table (List[List]): Table to convert.

        Returns:
            str: Formatted string representation of the table.
        """
        num_columns = len(table[0])
        column_widths = [max(len(str(row[i])) for row in table) for i in range(num_columns)]

        formatted_rows = []
        for row in table:
            formatted_row = [str(row[i]).ljust(column_widths[i]) for i in range(num_columns)]
            formatted_rows.append('  '.join(formatted_row))

        return '\n'.join(formatted_rows)

    def metric_as_str(self, metric, tex_format=False):
        """Get the metric as a formatted string.

        Args:
            metric: Metric to format.
            tex_format (bool, optional): Whether to format as LaTeX. Defaults to False.

        Returns:
            str: Formatted string representation of the metric.
        """
        return CrossValidationCompiler.str_format(self._score_dict[str(metric)],
                                                    np.mean(self._score_dict['n_samples']),
                                                    tex_format)

    def as_str(self, tex_format=False):
        """Get the compiled results as a formatted string.

        Args:
            tex_format (bool, optional): Whether to format as LaTeX. Defaults to False.

        Returns:
            str: Formatted string representation of the compiled results.
        """
        ret = ['' for _ in self._score_dict['metrics']]
        for i, metric in enumerate(self._score_dict['metrics']):
            ret[i] = self.metric_as_str(metric, tex_format)
        return ret

    def __str__(self) -> str:
        ret = ''
        for i, metric in enumerate(self._score_dict['metrics']):
            ret = f'{ret}{metric}[{self.metric_as_str(metric)}], '
        return ret[:-2]

class GridCompiler():
    """Class for compiling grid search results.

    This class compiles the results of grid search evaluations, including metric scores for each
    combination of parameters and each metric, and provides methods for formatting the results.
    """
    default_metric_list = [Metric.SP_INDEX,
                           Metric.BALANCED_ACCURACY,
                           Metric.MACRO_F1,
                           Metric.MACRO_RECALL,
                           Metric.MICRO_RECALL,
                           Metric.MACRO_PRECISION,
                           Metric.MICRO_PRECISION]

    def __init__(self,
                 metric_list: typing.List[Metric] = default_metric_list,
                 comparison_test: Test = None):
        self.cv_dict = {}
        self.param_dict = {}
        self.params = None
        self.metric_list = metric_list
        self.comparison_test = comparison_test

    def is_empty(self) -> bool:
        return self.params == None

    @staticmethod
    def calc_hash(params: typing.Dict) -> str:
        correct_params = {}
        for key, value in params.items():
            if isinstance(value, list):
                correct_params[key] = ', '.join([str(v) for v in value])
            else:
                correct_params[key] = value

        return hash(tuple(correct_params.items()))

    def add(self,
            params: typing.Dict,
            i_fold: int,
            target: typing.Iterable[int],
            prediction: typing.Iterable[int]) -> None:
        """
        Add evaluation results for a specific combination of parameters and fold.

        Args:
            grid_id (str): Identifier for the grid search combination of parameters.
            i_fold (int): Index of the fold.
            target (Iterable[int]): True labels.
            prediction (Iterable[int]): Predicted labels.
        """
        params_hash = GridCompiler.calc_hash(params)
        self.params = params.keys()

        if not params_hash in self.cv_dict:
            self.cv_dict[params_hash]  = {
                'params': params,
                'cv': CrossValidationCompiler(),
            }
            self.param_dict[params_hash] = params

        self.cv_dict[params_hash]['cv'].add(i_fold = i_fold,
                                metric_list = self.metric_list,
                                target = target,
                                prediction = prediction)

    def add_cv(self,
            params: typing.Dict,
            cv: CrossValidationCompiler):
        params_hash = GridCompiler.calc_hash(params)
        self.params = params.keys()

        if not params_hash in self.cv_dict:
            self.param_dict[params_hash] = params

        self.cv_dict[params_hash]  = {
            'params': params,
            'cv': cv,
        }

    def to_df(self, tex_format=False) -> pd.DataFrame:

        if self.is_empty():
            return pd.DataFrame()

        headers = list(self.params)
        for metric in self.metric_list:
            headers.append(metric.as_label())

        df = pd.DataFrame(columns=headers)

        for _, cv_dict in self.cv_dict.items():
            
            line = []

            for _, param_value in cv_dict['params'].items():
                line.append(str(param_value))

            for metric in self.metric_list:
                line.append(cv_dict['cv'].metric_as_str(metric, tex_format=tex_format))

            df.loc[len(df)] = line

        return df
    
    def get_metric_list(self, params: typing.Dict, metric: Metric):
        params_hash = GridCompiler.calc_hash(params)
        return self.cv_dict[params_hash]['cv'].get(metric)
    
    def get_best(self, metric: Metric = Metric.SP_INDEX):
        if self.is_empty():
            return [], []

        best_cv = None
        best_mean = 0

        for _, cv_dict in self.cv_dict.items():
            mean = np.mean(cv_dict['cv'].get(metric))
            if best_mean < mean:
                best_mean = mean
                best_cv = cv_dict

        return best_cv['params'], best_cv['cv']

    def get_cv(self, params: typing.Dict):
        params_hash = GridCompiler.calc_hash(params)
        return self.cv_dict[params_hash]['cv']

    def get_param_by_index(self, index: int):
        return self.cv_dict[list(self.param_dict.keys())[index]]['params']

    def get_cv_by_index(self, index: int):
        return self.cv_dict[list(self.param_dict.keys())[index]]['cv']

    def print_best_cm(self, filename: str = None, metric: Metric = Metric.SP_INDEX, relative=True):
        if self.is_empty():
            return

        _, cv = self.get_best(metric=metric)
        cv.print_cm(filename=filename, relative=relative)

    def export(self, filename):
        if self.is_empty():
            print('Grid not exported - empty grid')
            return

        file_extension = filename.split('.')[-1]

        if file_extension == "csv":
            df = self.to_df()
            df.to_csv(filename, index=False)

        elif file_extension == "tex":
            df = self.to_df()
            df.to_latex(filename, index=False)

        elif file_extension == "pkl":
            with open(filename, 'wb') as f:
                dill.dump(self, f)

        else:
            raise NotImplementedError(f'File extension not suported {file_extension}')

    @staticmethod
    def load(filename: str) -> 'GridCompiler':
        with open(filename, 'rb') as f:
            return dill.load(f)

    def as_table(self, tex_format=False) -> typing.List[typing.List[str]]:
        if self.is_empty():
            return 'Empty grid'

        """
        Get the compiled results as a formatted table.

        Args:
            tex_format (bool, optional): Whether to format the table for LaTeX. Defaults to False.

        Returns:
            List[List[str]]: Formatted table representation of the compiled results.
        """
        table = [[''] * (len(self.params) + len(self.metric_list)) for _ in range(len(self.cv_dict)+1)]

        j = 0
        for param in self.params:
            table[0][j] = str(param).replace('_', ' ')
            j = j + 1

        for metric in self.metric_list:
            table[0][j] = metric.as_label()
            j += 1

        i = 1
        for _, cv_dict in self.cv_dict.items():
            j = 0

            for _, param_value in cv_dict['params'].items():
                table[i][j] = str(param_value)
                j = j+1

            for metric in self.metric_list:
                table[i][j] = cv_dict['cv'].metric_as_str(metric, tex_format=tex_format)
                j += 1

            i += 1

        return table

    def as_str(self, tex_format=False):
        if self.is_empty():
            return 'Empty grid'
        """
        Get the compiled results as a formatted string.

        Args:
            tex_format (bool, optional): Whether to format the string for LaTeX. Defaults to False.

        Returns:
            str: Formatted string representation of the compiled results.
        """
        return CrossValidationCompiler.table_to_str(
            self.as_table(tex_format=tex_format)
        )

    def print_cm(self):
        if self.is_empty():
            return 'Empty grid'

        ret = '------- Confusion Matrix -------------\n'

        for hash, dict in self.cv_dict.items():
            ret += f'-- {dict["params"]} --\n\n'
            ret += dict['cv'].print_abs_cm()
            ret += '\n'
            ret += dict['cv'].print_rel_cm()
            ret += '\n'
        return ret

    def __str__(self) -> str:

        if self.is_empty():
            return 'Empty grid'

        ret = '------- Metric Table -------------\n'
        ret += self.as_str()
        return ret