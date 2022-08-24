import tensorflow as tf
import tensorflow_datasets as tfds
from sbr.datasets.structured import gtex
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def dataset_setup(sample_count_threshold=500, test_fraction = 0.1, validation_fraction = 0.1, verbose = True, batch_size = 32, seed = None):
    ###
    # Retrieve and prep the data
    ###

    # load X, y, class_names
    ds, info = tfds.load("gtex", split="train", with_info = True, as_supervised=True)
    l=list(iter(ds.take(info.splits['train'].num_examples)))
    X, y = map(np.array, zip(*l))
    class_names=np.array(info.features['target'].names)

    # xxx If you're going to drop anything, do it here (and adjust class_names):
    import pandas as pd
    counts={}
    y_class_names=pd.DataFrame([class_names[c] for c in y])
    for label in class_names:
        count = y_class_names[y_class_names[0]==label].count().squeeze()
        if sample_count_threshold is not None and count < sample_count_threshold:
            print(f"too few: {label}({count})")
        else:
            counts[label]=count
    print("Counts per class:")
    print(f"(total number of samples, total number of genes) = \n{X.shape}")
    for idx, (label, count) in enumerate(counts.items()):
        print(f"[{idx:2}] {label:12}\t{count:4} samples")

    # Normalize X and one-hot-encode y
    X=np.log2(np.array(X)+1)
    X=X/X.max()
    y=tf.one_hot(y, len(class_names))

    ### 
    # Split the data
    ###
    batch_size=32
    from sbr.preprocessing.dataset import multicategorical_split,  trim_list_size_to_batch_size_factor
    # only select this many samples: SAMPLE_COUNT_THRESHOLD * len(class_names)
    # but nothing is done here to balance them, so no classes are dropped here.
    x_train, y_train, x_validation, y_validation, x_test, y_test = multicategorical_split(X, y, 
                                                                                          sample_count_threshold = sample_count_threshold,
                                                                                          test_fraction = test_fraction, 
                                                                                          validation_fraction = validation_fraction,
                                                                                          verbose=verbose, 
                                                                                          batch_size=batch_size,
                                                                                          seed=seed)

    [x_train, y_train, x_validation, y_validation, x_test,y_test] = trim_list_size_to_batch_size_factor(batch_size, [x_train, y_train, 
                                                                                                       x_validation, 
                                                                                                       y_validation, 
                                                                                                       x_test, y_test])
    return [class_names, x_train, y_train, x_validation, y_validation, x_test,y_test]


def dataset_setup_orig(sample_count_threshold=100,
                  expr_path = "data/gtex/expr.ftr",
                  attr_path = 'dist/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt',
                  drop_classes_list = None,
                  attr_class_name_column_name = "SMTS",
                  attr_sample_id_column_name = "SAMPID",
                  expr_sample_id_column_name = "sample_id",
                  verbose=True):
    """Reads the expression and attribute feather files, normalizes the expression values, one-hot encodes the classes, and returns the features, targets and labels in coordinated order.

    Args:
      sample_count_threshold: drop any classes that are less than this threshold. If 'None', don't drop any classes
      expr_path: 
      attr_path: 
      drop_classes_list:
      attr_class_name_column_name:
      attr_sample_id_column_name:
      expr_sample_id_column_name:
      verbose: 
    
    Returns:
      (X, y, class_names, label_df)

      * X: Normalized feature values
      * y: One-hot encoded target values
      * class_names: Ordered list of strings, one item per class. This will be handy for understanding the predictions
      * label_df: a dataframe that combines X, y, class_names, and IDs together

    Example Usage;
      >>> X, y, class_names, label_df = dataset_setup(100)
      >>> # The return from this function (X, y) can be split as such:
      >>> x_train, x_test,y_train, y_test = sklearn.model_selection.train_test_split(X, np.array(y), test_size=1.-fraction, random_state=42, shuffle=True)`
      >>> # Class names can be retrieved from the returned target aray (y) and ordered list of class names (class_names) as such:
      >>> class_names[np.argmax[y]]

    
    """
    expr_df = pd.read_feather(expr_path)
    attr_df = pd.read_table(attr_path)
    label_df=attr_df[[attr_sample_id_column_name,attr_class_name_column_name]].merge(expr_df, how='inner', left_on=attr_sample_id_column_name, right_on=expr_sample_id_column_name)
    if drop_classes_list == None:
        labels = label_df.SMTS.unique()
    else:
        labels = [s for s in label_df.SMTS.unique() if s not in drop_classes_list]
    counts={}
    if sample_count_threshold != None:
        if verbose:
            print(f"Drop under-represented classes with less than {sample_count_threshold} samples:")
    for label in label_df[attr_class_name_column_name].unique():
        count = label_df[(label_df[attr_class_name_column_name]==label)].shape[0]
        if sample_count_threshold != None and count < sample_count_threshold:
            label_df = label_df[label_df[attr_class_name_column_name]!=label]
            if verbose:
                print(f"dropped {label}")
        else:
            counts[label]=count
    if verbose:
        print("Counts per class:")
        print(f"(total number of samples, total number of genes) = \n{label_df.shape}")
    for idx, (label, count) in enumerate(counts.items()):
        if verbose:
            print(f"[{idx:2}] {label:12}\t{count:4} samples")
    labels=list(label_df[attr_class_name_column_name].unique())
    
    X=label_df.drop([attr_class_name_column_name, attr_sample_id_column_name, expr_sample_id_column_name],axis=1)

    # normalize X
    X=np.log2(np.array(X)+1)
    X=X/X.max()

    # one-hot encode y
    le = LabelEncoder()
    y = tf.one_hot(le.fit_transform(list(label_df[attr_class_name_column_name])), len(labels))
    # labels order is not retained in the labeleoncoder, so reset the order in class_names for use later;
    # given an index, class_names will tell you the tissue type
    class_names = le.classes_

    return X, y, class_names, label_df
