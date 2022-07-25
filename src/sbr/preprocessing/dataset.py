import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


def multicategorical_split(X, y,
                           sample_count_threshold = 100,
                           test_fraction = 0.1,
                           validation_fraction = 0.1,
                           verbose = True,
                           batch_size=32,
                           seed=None,
                           shuffle = True):
    """
    Shuffles and splits X, y into test, train, validate; round dataset sizes to be a factor of batch_size.
    
    Final dataset size is (sample_count_threshold * <number of classes>) 
 
    see also: sbr.preprocessing.gtex.dataset_setup

    Args:
      X: Features
      y: multicategorical targets (more than one column)
      sample_count_threshold: use about this many samples from each class
      seed[None]: set this to make function deterministic/repeatable
      shuffle[True]: probably don't touch this. Shuffling the data really helps down-stream model training.

    Returns:
      (x_train, y_train, x_val, y_val, x_test, y_test)

    """
    
    if sample_count_threshold > 0:
        subsample_fraction=sample_count_threshold*len(np.unique(np.argmax(y,axis=1)))/X.shape[0]
    else:
        subsample_fraction = 0

    x_subsampled_train, _, y_subsampled_train, _ = train_test_split(X, np.array(y), test_size=(1.0-subsample_fraction), random_state=seed, shuffle=True)
    x_tmptrain, x_test, y_tmptrain, y_test = train_test_split(x_subsampled_train, y_subsampled_train, test_size=(test_fraction), random_state=seed, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_tmptrain, y_tmptrain, test_size=validation_fraction/(1.-test_fraction), random_state=seed, shuffle=True)
    if verbose:
        print("")
        print(f"Total remaining samples after subsampling= {x_train.shape[0] + x_val.shape[0] + x_test.shape[0]}")

    return x_train, y_train, x_val, y_val, x_test, y_test

# truncate sample size to be a factor of the batch size
def trim_list_size_to_batch_size_factor(batch_size=32, trim_list=None):
    """
    Trims the given list of multicategorical arrays down to a factor
    of the given batch_size. This can avoid errors during training
    when the dataset is very large, a small amount of data loss isn't
    a factor, and retaining a specfic batch_size (e.g., of 32) is
    prefered .

    Args:
      trim_list: a list of arrays to be trimmed
      batch_size[32]: probably leave this alone

    Returns:
      the same trim_list, but trimmed

    Usage:
      [x_train, y_train, x_val, y_val, x_test, y_test] = trim_list_size_to_batch_size_factor([x_train, y_train, x_val, y_val, x_test, y_test])

    """
    return [ x[:-( x.shape[0] % batch_size ), :] if x.shape[0] % batch_size != 0 else x for x in trim_list ]

