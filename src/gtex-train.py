"""
 Train a model that can be used for classifying tissue type given a gene expression profile.

 FIRST: run this exactly one time to set up the gtex dataset:
 >>> tfds build --register_checksums --overwrite  sbr/datasets/structured/gtex
"""

###
# Get the dataset
###
from sbr.preprocessing import gtex
[class_names,
 x_train, y_train, x_validation, y_validation, x_test,y_test] = gtex.dataset_setup(sample_count_threshold=500, 
                                                                                   test_fraction = 0.1, 
                                                                                   validation_fraction = 0.1, 
                                                                                   verbose = True, 
                                                                                   batch_size = 32, 
                                                                                   seed = None)

####
# Compile the model
####
from sbr import compile
specificityAtSensitivityThreshold=0.50
sensitivityAtSpecificityThreshold=0.50
model = compile.one_layer_multicategorical(input_size=x_train.shape[1],
                                           output_size=y_train.shape[1],
                                           dim=1000,
                                           output_activation='softmax',
                                           learning_rate=0.0001,
                                           isMultilabel=True,
                                           specificityAtSensitivityThreshold=specificityAtSensitivityThreshold,
                                           sensitivityAtSpecificityThreshold=sensitivityAtSpecificityThreshold,
                                           verbose=True)

####
# Fit the model (train)
####
from sbr import fit
history=fit.multicategorical_model(model=model, 
                                   model_folder ='data/model/gtex',
                                   x_train=x_train, y_train=y_train, 
                                   x_validation=x_validation, y_validation=y_validation, 
                                   epochs = 200,
                                   patience = 4,
                                   lr_patience = 2,
                                   checkpoint_verbose=1,
                                   train_verbose=0)
####
# EVALUATE
####

# sanity check fitting
from sbr.evaluate import training_report
performance = training_report(model, x_test, y_test, 
                              sensitivityAtSpecificityThreshold=sensitivityAtSpecificityThreshold,
                              specificityAtSensitivityThreshold=specificityAtSensitivityThreshold,
                              verbose=True)
    
# visualize training performance
from sbr.visualize import plot_loss_curve
plot_loss_curve(history=history, 
                figsize=(5,5),
                metrics = ['loss','accuracy','val_accuracy'],
                write_directory="data/images",
                file_name="nn-loss-curve.png",
                show_plot=True)

# check predictions
print(f"Number of training samples: {len(y_train)}")
print(f"Number of validation samples: {len(y_validation)}")
from sbr.evaluate import compare_predictions
_, _ = compare_predictions(model=model, x_test=x_test, y_test=y_test.astype(int), 
                           class_names=class_names, 
                           verbose = True)

print("Saving architecture...")
# everything OK? save it.
from sbr.model import save_architecture
if save_architecture(model=model, model_path='data/model/gtex/manual', file_name="gtex_model.h5", input_size = x_train.shape[1], verbose=True):
    print("Done.")
else:
    print("ERROR: Something went wrong with saving the model file.")
    
