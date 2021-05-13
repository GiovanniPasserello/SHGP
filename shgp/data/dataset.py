from typing import List, Optional

import numpy as np
import os
import re

from shgp.data.utils import standardise_features

"""
General
"""


class Dataset:
    """
        A dataset utilities class for loading data and storing metadata.
    """

    def __init__(
        self,
        name: str,
        filename: str,
        *,
        delimiter: Optional[str] = ' ',
        skiprows: Optional[int] = 0,
        x_slice: Optional[np.lib.index_tricks.IndexExpression] = np.s_[:, :-1],
        y_slice: Optional[np.lib.index_tricks.IndexExpression] = np.s_[:, -1],
        x_delete_columns: Optional[List[int]] = None
    ):
        self.name = name
        absolute_path = re.findall('.*/shgp/', os.getcwd())[0]
        self.path = absolute_path + 'data/datasets/{}'.format(filename)
        self.delimiter = delimiter
        self.skiprows = skiprows
        self.x_slice = x_slice
        self.y_slice = y_slice
        self.x_delete_columns = x_delete_columns

    def load_data(self):
        data = np.loadtxt(self.path, delimiter=self.delimiter, skiprows=self.skiprows)
        X = standardise_features(data[self.x_slice])
        Y = data[self.y_slice].reshape(-1, 1)

        # Remove specified column indices.
        # The indices should correspond to the columns *after* splitting (X, Y).
        if self.x_delete_columns:
            X = np.delete(X, self.x_delete_columns, axis=1)

        # Shuffle the dataset to avoid training bias.
        permutation = np.random.permutation(len(X))

        return X[permutation], Y[permutation]

    # TODO: The data should only be standardised, w.r.t. the training set parameters
    #       Need to change this so that the test set is standardised after.
    def load_train_test_split(self, train_proportion=0.9):
        assert 0.0 < train_proportion < 1.0, 'train_proportion must be: 0.0 < X < 1.0'

        X, Y = self.load_data()
        N = len(X)
        train_size = int(N * train_proportion)
        permutation = np.random.permutation(N)

        train_split, test_split = permutation[:train_size], permutation[train_size:]
        X_train, Y_train = X[train_split], Y[train_split]
        X_test, Y_test = X[test_split], Y[test_split]

        return X_train, Y_train, X_test, Y_test


"""
Toy Datasets
"""


class PlatformDataset(Dataset):
    # https://github.com/GPflow/docs/tree/master/doc/source/notebooks/basics/data
    # N=50, D=1, C=2

    def __init__(self):
        super().__init__(
            name='Platform',
            filename='toy/platform.csv',
            delimiter=','
        )


class BananaDataset(Dataset):
    # https://github.com/GPflow/docs/tree/master/doc/source/notebooks/basics/data
    # N=400, D=2, C=2

    def __init__(self):
        super().__init__(
            name='Banana',
            filename='toy/banana.csv',
            delimiter=','
        )


"""
Real Datasets
"""


class FertilityDataset(Dataset):
    # https://archive.ics.uci.edu/ml/datasets/Fertility
    # N=100, D=9, C=2

    def __init__(self):
        super().__init__(
            name='Fertility',
            filename='fertility.txt',
            delimiter=','
        )


class CrabsDataset(Dataset):
    # https://datarepository.wolframcloud.com/resources/Sample-Data-Crab-Measures
    # N=200, D=6, C=2

    def __init__(self):
        super().__init__(
            name='Crabs',
            filename='crabs.csv',
            delimiter=',',
            skiprows=1,
            x_slice=np.s_[:, 1:],
            y_slice=np.s_[:, 0],
            x_delete_columns=[1]  # index column
        )


class HeartDataset(Dataset):
    # https://www.openml.org/d/53
    # N=270, D=13, C=2

    def __init__(self):
        super().__init__(
            name='Heart Statlog',
            filename='heart.csv',
            delimiter=',',
            skiprows=1
        )


class IonosphereDataset(Dataset):
    # https://archive.ics.uci.edu/ml/datasets/ionosphere
    # N=351, D=33, C=2

    def __init__(self):
        super().__init__(
            name='Ionosphere',
            filename='ionosphere.txt',
            delimiter=',',
            x_delete_columns=[1]  # column of zeros
        )


class BreastCancerDataset(Dataset):
    # https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
    # N=569, D=30, C=2

    def __init__(self):
        super().__init__(
            name='Breast Cancer',
            filename='breast-cancer-diagnostic.txt',
            delimiter=',',
            x_slice=np.s_[:, 2:],
            y_slice=np.s_[:, 1]
        )


class PimaDataset(Dataset):
    # http://networkrepository.com/pima-indians-diabetes.php
    # N=768, D=8, C=2

    def __init__(self):
        super().__init__(
            name='Pima Diabetes',
            filename='pima-diabetes.csv',
            delimiter=',',
        )


class TwonormDataset(Dataset):
    # https://www.openml.org/d/1507
    # N=7400, D=20, C=2

    def __init__(self):
        super().__init__(
            name='Twonorm',
            filename='twonorm.csv',
            delimiter=',',
            skiprows=1
        )

    def load_data(self):
        X, Y = super().load_data()
        return X, Y - 1  # the labels must be in [-1, 1] or [0, 1], not [1, 2]


class RingnormDataset(Dataset):
    # https://www.openml.org/d/1496
    # N=7400, D=20, C=2

    def __init__(self):
        super().__init__(
            name='Ringnorm',
            filename='ringnorm.csv',
            delimiter=',',
            skiprows=1
        )

    def load_data(self):
        X, Y = super().load_data()
        return X, Y - 1  # the labels must be in [-1, 1] or [0, 1], not [1, 2]


class MagicDataset(Dataset):
    # https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
    # N=19020, D=10, C=2

    def __init__(self):
        super().__init__(
            name='MAGIC Telescope',
            filename='magic.txt',
            delimiter=','
        )


class ElectricityDataset(Dataset):
    # https://datahub.io/machine-learning/electricity
    # N=45312, D=8, C=2

    def __init__(self):
        super().__init__(
            name='Electricity',
            filename='electricity.csv',
            delimiter=',',
            skiprows=1
        )


# Test
if __name__ == '__main__':
    ds = ElectricityDataset()
    X, Y = ds.load_data()
    pass
