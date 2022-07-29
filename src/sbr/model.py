import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
def save_architecture(model, model_path: str = None, file_name = "model.h5", input_size = None, verbose = 1):
    """
    Saves the given model to the given path and name. It's a good idea
    to train and then run this in a notebook if possible so the train
    model is resident in memory because this function can be tried
    again in case it fails for some reason.

    .. note:: 
    
      Custom layer BADBlock will be loaded as part of the configuration.

    .. warning:: 

      THIS WILL OVER-WRITE ANY EXISTING MODEL.



    Args:
    
      model: model object for calling `model.save`

      model_path: file path where model is to be written

      file_name: name of the file, h5 format. Any exisiting file will be over-written.

      input_size: if not None, attempts to check predictions on saved model are close to original model

      verbose: 0: debug, 1:print out model summary. This may throw an error if model wasn't compiled with a known input size

    Returns:
      True on success, False otherwise. Check the return to try again if it fails while model is still resident in memory.

    Example usage:

      >>> success = sbf.model.save(model, model_path="data/model/manual", file_name="model.h5", verbose=1)
      True

    """
    func_name = "[sbr.model.save_architecture]"
    try: 
        if model_path is None:
            print(f"! {func_name} ERROR: no model_path provided.")
            return False
        os.makedirs(model_path, exist_ok=True)
        full_path = os.path.join(model_path, file_name)
        model.save(full_path)

        if verbose == 0:
            print(f"{func_name} Loading model back in for a check...")

        from sbr.layers import BADBlock
        savedModel = tf.keras.models.load_model(full_path, custom_objects={'BADBlock': BADBlock})

        if verbose == 0:
            print(f"{func_name} using model to do a prediction and comparing to original model...")
        # extra check
        if input_size is not None:
            x = tf.random.uniform((10, input_size))
            orig = model.predict(x)
            saved = savedModel.predict(x)
            if verbose == 0:
                print(f"{func_name} (orig model prediction), (saved model prediction) = \n{np.argmax(orig,axis=1)}\n{np.argmax(saved, axis=1)}")
            assert np.allclose(np.argmax(orig, axis=1), np.argmax(saved,axis=1))

        if verbose == 1:
            print(f"{func_name} Model successfully saved at: {full_path}.")
            savedModel.summary()

        return True

    except Exception as e:
        import traceback
        import sys
        # its really important to not just exit the current execution state if the model save fails after hours of training, 
        # so catch and print out exception issues if it fails.
        print(f"! {func_name} ERROR: model not saved. Exception ({type(e)}) : {e}")
        print(f"! {func_name}    model_path={model_path}, file_name={file_name}, full_path={full_path}, stack={traceback.print_exception(*(sys.exc_info()))}")
        return False
