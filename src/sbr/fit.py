import numpy as np
import tensorflow as tf
import sklearn

def multicategorical_model(model, model_folder, x_train, y_train, x_validation, y_validation,
                           epochs=200,
                           patience=4,
                           lr_patience=2,
                           lr_factor=0.1,
                           batch_size=32,
                           shuffle_value=100,
                           initial_epoch=0,
                           train_verbose=1,
                           checkpoint_verbose=1):
    """Fits the given model with the given hyperparameters and multi-categorical data, after computing class weights and shuffling the data. Writes checkpoint and final model weights to `model_folder`. Look under `variables/variables.*` for weights. 

    .. admonition:: Assumptions
    
        * Model has been compiled and saved to f"{model_path}.h5" (e.g., `data/model/gtex/manual/gtex_model.h5`)
        * Targets are one-hot encoded
        * Features have been normalized

    Tested with tensorflow v2.6.2, keras 2.6.0
    
    Args:
      model: a compiled model

      model_folder: writable folder to store the checkpoint and final model weights

      x_train: training features, see sbr.split for help

      y_train: training targets,  see above

      x_validation: validation feature, see above

      y_validation: validation feature, see above

      epochs [200]: Number of epochs to train

      patience [4]: Number of epochs with no improvement after which training will be stopped.

      lr_patience [2]: Number of epochs with no improvement after which learning rate will be reduced.

      lr_factor [0.1]: Factor by which the learning rate will be reduced. new_lr = lr * factor.

      batch_size [32]: probably don't change this

      shuffle_value [100]:

      initial_epoch [0]: use this if you want to resume training at a particular epoch

      train_verbose [0]: amount of information to print on each epoch. for 0: silent, 1: animated progress bar, 2: mentions epoch. For example:

        * 0: <silent>
        * 1: 
        .. code-block::

          [==================]
          Epoch 00015: val_loss improved from 0.06645 to 0.06611, saving model to 
               data/model/gtex
          INFO:tensorflow:Assets written to: data/model/gtex/assets
    

        * 2: 

        .. code-block::

          Epoch 1/10
          checkpoint_verbose [1]: amount of information to print on each epoch about the 
              checkpoint. 0: silent.



    Returns: 
      history

      A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable). Use `print(history.history.keys())` to see all the hist and `print(history.history['val_loss'])` to print validation loss

    Example Usage:
      >>> from sbr import fit
      >>> history=fit.multicategorical_model(model=model, 
                                   model_folder ='data/model/gtex',
                                   x_train=x_train, y_train=y_train, 
                                   x_validation=x_validation, y_validation=y_validation, 
                                   epochs = 200,
                                   patience = 4,
                                   lr_patience = 2,
                                   checkpoint_verbose=1,
                                   train_verbose=0)


    Example Usage: Reload with: 
      >>> model = load_model('f{model_path}')
      >>> model.load_weights(f"{model_folder}")




    """
    # compute class weights
    # bigger number reflects fewer samples
    from sklearn.utils.class_weight import compute_class_weight
    y_integers = np.argmax(y_train, axis=1)
    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', 
                                                                    classes=np.unique(y_integers), 
                                                                    y=y_integers)
    class_weights = dict(enumerate(class_weights))

    
    # set up the callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    checkpoint = ModelCheckpoint(model_folder, monitor='val_loss', verbose=checkpoint_verbose, 
                                 save_best_only=True, mode='min', save_freq='epoch')
    earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=patience)
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience)

    # setup data for best performance
    train_data=tf.data.Dataset.from_tensor_slices((x_validation,y_validation))
    train_data=train_data.repeat().shuffle(shuffle_value).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # fit the model and return the history
    return model.fit(x_train, y_train, 
        steps_per_epoch=y_train.shape[0]/batch_size,
        batch_size=batch_size, 
        epochs=epochs, 
        initial_epoch= initial_epoch, 
        validation_data=(x_validation, y_validation),
        callbacks=[earlystop, reduce, checkpoint],
        class_weight = class_weights,
        verbose=train_verbose
        )

