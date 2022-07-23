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
      (y_pred, pairs) where
      y_pred: the predicted outcomes from x_test
      pairs: list pairs of (<truth><false-prediction>) class names

    Exampe usasge:
      y_pred, pairs = compare_predictions(model=model, x_test=x_test, y_test=ytest, class_names=class_names, verbose = True)
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

def report(model, x_test, y_test, 
           sensitivityAtSpecificityThreshold=None,
           specificityAtSensitivityThreshold=None,
           verbose=True):
    """
    Calls `model.evaluate(x_test,y_test)` and, if verbose, reports on the performance, then returns a performance object.

    Args:
      x_test: features
      y_test: targets
      verbose: if True, report to stdout
      sensitivityAtSpecificityThreshold: If not None, and verbose, and this metric was captured in model.fit, report it to stdout
      specificityAtSensitivityThreshold: see above

    Returns:
      performance

    Example useage:
      
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
        '''
        print(f"Loss:\t{performance[0]}")
        print(f"Accuracy:\t{performance[1]}")
        print(f"MSE:\t{performance[2]}")
        print(f"Precision:\t{performance[3]}")
        print(f"Recall:\t{performance[4]}")
        print(f"AUC:\t{performance[5]}")
        print(f"specificityAtSensitivity,threshold={specificityAtSensitivityThreshold}:\t{performance[6]}")
        print(f"Sensitivityatspecificity,threshold={sensitivityAtSpecificityThreshold}:\t{performance[7]}")
        print(f"False positives:\t{performance[8]}")
        print(f"False negatives:\t{performance[9]}")
        print(f"True positives:\t{performance[10]}")
        print(f"True negatives:\t{performance[11]}")
        '''
    return performance
