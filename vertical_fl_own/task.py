from pathlib import Path
from logging import WARN
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn as nn
from flwr.common.logger import log

from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner

NUM_VERTICAL_SPLITS = 3


# this version: all columns are cleaned, Each farm and vet get an average antibiotic use (to approach random effect
# in the original model)


def _bin_houses(NumberOfHouses_series):
    bins = [-np.inf, 1.01, 2.01, 3.01, np.inf]
    labels = ["1", "2", "3", "morethan3"]
    return (
        pd.cut(NumberOfHouses_series, bins=bins, labels=labels, right=True)
        .astype(str)
        .replace("nan", "Unknown")
    )


def _create_features(df):
    # Convert 'number of houses' to numeric, coercing errors to NaN
    df["NumberOfHouses"] = pd.to_numeric(df["NumberOfHouses"], errors="coerce")
    df["NumberOfHouses"] = _bin_houses(df["NumberOfHouses"])
    df.drop(columns=["FarmIdentification", "VetId"], inplace=True)   # I have no "Id" column now!
    all_keywords = set(df.columns)

    # make dummy variables and standardize the rest of the columns
    categorical_columns = ["NumberOfHouses", "HatchQuarter", "Type"]
    other_columns = [col for col in df.columns if col not in categorical_columns
                     and col != "AntibioticsAfterWeek1"]

    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    df = df.astype({col: 'int' for col in df.columns if df[col].dtype == 'bool'})
    df[other_columns] = (df[other_columns] - df[other_columns].mean()) / df[other_columns].std()

    return df, all_keywords


def process_dataset():

    df = pd.read_csv("C:/Users/4243692/flower-vfa-deployment/vertical-fl-own/data/train.csv")
    processed_df = df.dropna().copy()
    # Remove free range and organic
    processed_df = processed_df[processed_df['Type'] != 'free range and organic']
    return _create_features(processed_df)


def load_data(partition_id: int, num_partitions: int):
    """Partition the data vertically and then horizontally.

    We create three sets of features representing three types of nodes participating in
    the federation.

    [{'HatchYear', 'FarmIdentification', 'Type', 'VetId'}, {'Thinning', 'EndFlockSize', 'HatchQuarter', 'NumberOfHouses'}, {'AntibioticsAfterWeek1'}]

    Once the whole dataset is split vertically and a set of features is selected based
    on mod(partition_id, 3), it is split horizontally into `ceil(num_partitions/3)`
    partitions. This function returns the partition with index `partition_id % 3`.
    """

    if num_partitions != NUM_VERTICAL_SPLITS:
        log(
            WARN,
            "To run this example with num_partitions other than 3, you need to update how "
            "the Vertical FL training is performed. This is because the shapes of the "
            "gradients might not be the same along the first dimension.",
        )

    # Read whole dataset and process
    processed_df, features_set = process_dataset()

    # Vertical Split and select
    v_partitions = _partition_data_vertically(processed_df, features_set)
    v_split_id = np.mod(partition_id, NUM_VERTICAL_SPLITS)
    v_partition = v_partitions[v_split_id]

    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(v_partition)

    # Split horizontally with Flower Dataset partitioner
    num_h_partitions = int(np.ceil(num_partitions / NUM_VERTICAL_SPLITS))
    partitioner = IidPartitioner(num_partitions=num_h_partitions)
    partitioner.dataset = dataset

    # Extract partition of the `ClientApp` calling this function
    partition = partitioner.load_partition(partition_id % num_h_partitions)
    partition.remove_columns(["AntibioticsAfterWeek1"])

    return partition.to_pandas(), v_split_id


def _partition_data_vertically(df, all_keywords):
    partitions = []
    keywords_sets = [
        {"HatchYear", "Type", "FarmAvgAbAfterWk1", "VetAvgAbAfterWk1"}, # client 1
        {"Thinning", "EndFlockSize"},  # client 2
        # the rest goes to client 3: "HatchQuarter", "NumberOfHouses"
    ]
    keywords_sets.append(all_keywords - keywords_sets[0] - keywords_sets[1])

    for keywords in keywords_sets:
        partitions.append(
            df[
                list(
                    {
                        col
                        for col in df.columns
                        for kw in keywords
                        if kw in col or "AntibioticsAfterWeek1" in col
                    }
                )
            ]
        )

    return partitions


