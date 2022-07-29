import numpy as np
def compare_predictions(model, x_test, y_test, class_names=None, verbose = True):
    '''
    Predicts y_test from x_test using model, then compares predictions with truth.

    Args:
      model: the model to use `model.predict`
      x_test: test features
      y_test: targets
      class_names: an ordered list of class name strings that map to the `np.argmax(y_test,axis=1)` indices in y_test. If none, class indices will be reported instead of strng names.
      verbose: if verbose, pairs are printed out (good if there aren't a lot of mislabeled predictions)

    Returns: 

       `(y_pred, pairs)`

       * y_pred: the predicted outcomes from x_test
       * pairs: list pairs of (<truth><false-prediction>) class names



    Exampe usage:
      >>> y_pred, pairs = compare_predictions(model=model, 
                                              x_test=x_test, y_test=ytest, 
                                              class_names=class_names, 
                                              verbose = True)

    .. code-block:: text

      Number of test samples: 256
      Mis-classifications:
      (<truth>,<false-prediction>)
      [('Esophagus', 'Blood Vessel'), ('Blood Vessel', 'Heart'), ('Adipose Tissue', 'Breast'), ('Salivary Gland', 'Esophagus')]
      [sbr.model.save_architecture] Model successfully saved at: data/model/gtex/manual/gtex_model.h5.
      Model: "sequential"


    '''
    y_pred = model.predict(x_test)

    if verbose:
        print("")
        print(f"Number of test samples: {len(y_test)}")
        print("Mis-classifications:")
        print("(<truth>,<false-prediction>)")

    predicted=np.argmax(y_pred, axis=1)
    observed=np.argmax(y_test,axis=1)
    if class_names is not None:
        pairs=list(zip(class_names[observed[(predicted != observed)]],class_names[predicted[(predicted != observed)]]))
    else:
        pairs=list(zip(observed[(predicted != observed)],predicted[(predicted != observed)]))

    if verbose:
        print(pairs)
        
    return y_pred, pairs

def training_report(model, x_test, y_test, 
                    sensitivityAtSpecificityThreshold=None,
                    specificityAtSensitivityThreshold=None,
                    verbose=True):
    """
    Calls model.evaluate(x_test,y_test) and, if verbose, reports on the performance, then returns a performance object like the one returned by `model.evaluate <https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate>`_.

    Args:
      x_test: features
      y_test: targets
      verbose: if True, report to stdout
      sensitivityAtSpecificityThreshold: If not None, and verbose, and this metric was captured in model.fit, report it to stdout
      specificityAtSensitivityThreshold: see above

    Returns:
      A performance object

    Example Usage:
      >>> performance = training_report(model, x_test, y_test, 
                                    sensitivityAtSpecificityThreshold=sensitivityAtSpecificityThreshold,
                                    specificityAtSensitivityThreshold=specificityAtSensitivityThreshold,
                                    verbose=True)

    .. code-block:: text

      Performance: 
      Performance details: 
        loss:0.07804308831691742
        accuracy:0.984375
        mse:0.0009617832256481051
        precision:0.984375
        recall:0.984375
        auc:0.9988833665847778
        SpecificityAtSensitivity:0.9998437762260437
        SensitivityAtSpecificity:0.99609375
        fp:4.0
        fn:4.0
        tp:252.0
        tn:6396.0
      Figure(500x500)
      Number of training samples: 2080
      Number of validation samples: 256

    ...


    """
    performance = model.evaluate(x_test,y_test)
    if(verbose):
        print(f"Performance: ")
        print("Performance details: ")
        for idx, name in enumerate(model.metrics_names):
            if sensitivityAtSpecificityThreshold is not None and name == "specificityAtSensitivity":
                print(f"specificityAtSensitivity,threshold={specificityAtSensitivityThreshold}:\t{performance[6]}")
            elif specificityAtSensitivityThreshold is not None and name == "sensitivityAtSpecificity":
                print(f"Sensitivityatspecificity,threshold={sensitivityAtSpecificityThreshold}:\t{performance[7]}")
            else:
                print(f"{name}:\t{performance[idx]}")
    return performance


import pandas as pd
import numpy as np

def mislabeled_pair_counts(model, X, y, class_names, sample_ids=None, batch_size=1500, verbose=False):
    """
    For multicategorical models: creates a table of observed,
    predicted class names for the mispredicted observations. This
    tends to use a lot of memory on multiple runs in a jupyter
    notebook, with tensorflow 2.6. May need to restart the kernel on
    second run. If resources continue to be a problem after restarting
    the kernel, reduce the batch_size.
    
    Assumes y_pred, y_obs are one-hot encoded and class_names matches the index predictions returned from np.argmax(y_pred)

    Args:
        model: used for model.predict
        X: feature values
        y: one-hot encoded true labels
        class_names: ordered list of class_names
        sample_ids: pass in this Series object to get back a table of pairs with their sample_ids
        batch_size: number of samples to process in each step (to keep from swamping memory)
        verbose: helps with debugging; messages each step/batch

    Returns:
      (pairs_counts, pair_id_map)

      * pairs_counts: Table with compound index 'observed','predicted' and one column, "counts", with the count of all the  
      * the samples in that observed/predicted mislabeled pair.
      * pair_id_map: None if sample_ids wasn't passed in, otherwise returns a table with columns observed, predicted, sample_id

    Example Usage: Get the mislabeled counts
       >>> mislabeled_counts, mislabeled = mislabeled_pair_counts(model=model, X=X, y=y, class_names=class_names, 
                                                                  sample_ids = pd.Series(label_df["sample_id"]),
                                                                  batch_size=500)
       >>> mislabeled_counts

    .. image:: images/mislabeled_counts.png


    Example Usage: Get the mislabeled samples
       >>> m=mislabeled.reset_index()
       >>> m[m['observed']=="Lung"]

    .. image:: images/mislabeled_lung.png



    """
    # break up the dataset, there's too many to predict at once; define a function to use for the iterations
    def mislabeled_pairs(class_names, y_pred, y_obs, unique_sample_indexes):
        pred_df = pd.DataFrame(class_names[np.argmax(y_pred,axis=1)], columns=["predicted"])
        obs_df = pd.DataFrame(class_names[np.argmax(y_obs,axis=1)], columns=["observed"])
        index_series = pd.Series(unique_sample_indexes)
        pred_obs_df = pd.concat([index_series.reset_index(), pred_df, obs_df], axis=1)
        return pred_obs_df[pred_obs_df["observed"]!=pred_obs_df['predicted']].sort_values(by='observed')

    step = batch_size
    # xxx may be a buc - batch_size can't be a modulo X.shape[0]?

    # add unique sample indexes to allow grouping/counting
    if sample_ids is None:
        unique_sample_indexes = pd.Series(list(range(X.shape[0])))
        if verbose:
            print("[mislabeled_pair_counts] sample_ids is None, using arbitrary ids instead")
    else:
        unique_sample_indexes = sample_ids
        if verbose:
            print("[mislabeled_pair_counts] using specified sample_ids")

    # first iteration is a special case
    if verbose:
        print(f"[mislabeled_table]: first=0:{step}")
    y_pred_1 = model.predict(X[0:step])

    m_prev = mislabeled_pairs(class_names, y_pred_1, y[0:step], unique_sample_indexes[0:step])
    if verbose:
        print(f"[mislabeled_table]: start={step},stop={int(np.floor(X.shape[0]/step))*step},step={step}")

    # build the table one batch at a time, to conserve memory
    for i in range(step, int(np.floor(X.shape[0]/step)*step), step):
        low = i
        hi = low + step
        if verbose:
            print(f"[mislabeled_table]: low={low},hi={hi}")
        y_pred_1 = model.predict(X[low:hi])
        m = mislabeled_pairs(class_names, y_pred_1, y[low:hi], unique_sample_indexes[low:hi])
        m_prev = pd.concat([m_prev, m], axis=0)

    # last iteration is a special case
    low = int(np.floor(X.shape[0]/step))*step
    if low != X.shape[0]:
        # there are still some batch_size - x samples left, finish it up:
        if verbose:
            print("[mislabeled_table]: last step")
            print(f"[mislabeled_table]: low={low}")
            y_pred_1 = model.predict(X[low:])
            m = mislabeled_pairs(class_names, y_pred_1, y[low:], unique_sample_indexes[low:])
            m_prev = pd.concat([m_prev, m], axis=0)

    m_prev = m_prev.drop(['index'], axis=1)

    # if no sample_ids Series was passed in, then just return 'Null' for pair_id_map,
    if sample_ids is None:
        # In the absense of sample_ids, a range of consecutive integers was used to simulate unique items
        # in order to allow for counting the unique pairs.
        # Therefore, just return None for the pair_id_map to avoid confusion.
        pair_id_map = None
    else:
        pair_id_map = m_prev

    # group the pairs and count the number of mislabeled observations in each pair
    pairs_counts = m_prev.groupby(['observed','predicted']).count().sort_values(by='observed')
    # now 'final' has one column, created by the "unique_sample_indexes" Series; name it "counts"
    pairs_counts.columns = ["counts"]

    return pairs_counts, pair_id_map
