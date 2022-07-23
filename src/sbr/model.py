import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
def save_architecture(model, model_path=None, file_name="model.h5", input_size=None, verbose=True):
    """
    Saves the given model to the given path and name. It's a good idea
    to train and then run this in a notebook if possible so the train
    model is resident in memory because this function can be tried
    again in case it fails for some reason.

    WARNING: THIS WILL OVER-WRITE ANY EXISTING MODEL.

    Args:
      model: model object for calling `model.save`
      model_path[None]: file path where model is to be written
      file_name["model.h5"]: name of the file, h5 format. Any exisiting file will be over-written.
      input_size: if not None, attempts to check predictions on saved model are close to original model
      verbose[True]: print out model summary. This may throw an error if model wasn't compiled with a known input size

    Returns:
      True on success, False otherwise. Check the return to try again if it fails while model is still resident in memory.

    Example usage:
      success = sbf.model.save(model, model_path="data/model/manual", file_name="model.h5", verbose=True)
    """

    try: 
        if model_path is None:
            print("! [sbr.model.save] ERROR: no model_path provided.")
            return False
        os.makedirs(model_path, exist_ok=True)
        full_path = os.path.join(model_path, file_name)
        model.save(full_path)
        savedModel=tf.keras.models.load_model(full_path)

        # extra check
        if input_size is not None:
            x = tf.random.uniform((10, input_size))
            assert np.allclose(model.predict(x), savedModel.predict(x))

        if verbose:
            print(f"Model successfully saved at: {full_path}.")
            savedModel.summary()

        return True

    except Exception as e:
        # its really important to not just exit the current execution state if the model save fails after hours of training, 
        # so catch and print out exception issues if it fails.
        print(f"! [sbr.model.save] ERROR: model not saved. Exception ({type(e)}) : {e}")
        print(f"! [sbr.model.save]    model_path={model_path}, file_name={file_name}, full_path={full_path}")
        return False
