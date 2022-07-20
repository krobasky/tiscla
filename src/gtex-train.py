import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
expr_df = pd.read_feather("data/gtex/expr.ftr")
attr_df = pd.read_table('dist/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt')
label_df=attr_df[['SAMPID','SMTS']].merge(expr_df, how='inner', left_on='SAMPID', right_on='sample_id')
labels = [s for s in label_df.SMTS.unique() if s not in ['Bone Marrow', 'Fallopian Tube', 'Cervix Uteri', 'Bladder','Skin']]
SAMPLE_COUNT_THRESHOLD=100
TRAINING_REINITIALIZATIONS=5
counts={}
print(f"Drop under-represented classes with less than {SAMPLE_COUNT_THRESHOLD} samples:")
for label in label_df.SMTS.unique():
    count = label_df[(label_df.SMTS==label)].shape[0]
    # if count < 200 or label == "Heart" or label == "Lung" or label == "Brain":
    if count < SAMPLE_COUNT_THRESHOLD:
        label_df = label_df[label_df.SMTS!=label]
        print(f"dropped {label}")
    else:
        counts[label]=count
#given an index, class_names will tell you the tissue type
print("Counts per 'super-class':")
class_names=np.array(list(counts.keys()))
print(f"(total number of samples, total number of genes) = \n{label_df.shape}")
for idx, (label, count) in enumerate(counts.items()):
    print(f"[{idx:2}] {label:12}\t{count:4} samples")
labels=list(label_df.SMTS.unique())
#labels
X=label_df.drop(['SMTS','SAMPID','sample_id'],axis=1)
X=np.log2(np.array(X)+1)
X=X/X.max()
le = LabelEncoder()
y = tf.one_hot(le.fit_transform(list(label_df['SMTS'])), len(labels))
# only use about 100 samples of each class:
fraction=SAMPLE_COUNT_THRESHOLD*len(labels)/X.shape[0]
# change fraction to 1. to use entire dataset
x_train, x_test,y_train, y_test = train_test_split(X, np.array(y), test_size=1.-fraction, random_state=42, shuffle=True)
x_train, x_test,y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42, shuffle=True)
x_train, x_validation,y_train, y_validation = train_test_split(x_train, y_train, test_size=0.1, random_state=42, shuffle=True)
print("")
print(f"Total remaining samples after subsampling= {x_train.shape[0] + x_validation.shape[0] + x_test.shape[0]}")
import sklearn
from sklearn.utils.class_weight import compute_class_weight
y_integers = np.argmax(y_train, axis=1)
class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', 
                                                                classes=np.unique(y_integers), 
                                                                y=y_integers)
d_class_weights = dict(enumerate(class_weights))
# bigger number reflects fewer samples
d_class_weights
batch_size=32

extra=(x_train.shape[0] % batch_size)
x_train = x_train[:-extra,:]
#--
extra=(x_test.shape[0] % batch_size)
x_test = x_test[:-extra,:]
#--
extra=(x_validation.shape[0] % batch_size)
x_validation = x_validation[:-extra,:]

extra=(y_train.shape[0] % batch_size)
y_train = y_train[:-extra,:]
#--
extra=(y_test.shape[0] % batch_size)
y_test = y_test[:-extra,:]
#--
extra=(y_validation.shape[0] % batch_size)
y_validation = y_validation[:-extra,:]

import tensorflow.keras as keras
from keras import Sequential
from keras.layers import  Dense, BatchNormalization, Activation, Dropout
output_size=y_train.shape[1]
input_size=x_train.shape[1]
output_activation='softmax'
learning_rate=0.0001
isMultilabel=True
dim=1000
specificityAtSensitivityThreshold=0.50
sensitivityAtSpecificityThreshold=0.50

# added HeNormal initializer to force diversity of outcomes between trainings
model = Sequential()
model.add(Dense(dim, input_dim=input_size, name="input", # xxx
    kernel_initializer=tf.keras.initializers.HeNormal(), 
    bias_initializer=tf.zeros_initializer()
    ))
model.add(BatchNormalization(name='1_decoder_batch_norm'))
model.add(Activation('relu', name="1_activation"))
model.add(Dropout(0.50, name='1_hidden_dropout'))
model.add(Dense(output_size, activation=output_activation, name = "output",
    kernel_initializer=tf.keras.initializers.HeNormal(), 
    bias_initializer=tf.zeros_initializer()
    ))

model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate),
              metrics=['accuracy', 'mse',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.AUC(name='auc',
                                            multi_label=isMultilabel),
                       tf.keras.metrics.SpecificityAtSensitivity(specificityAtSensitivityThreshold, name='SpecificityAtSensitivity'),
                       tf.keras.metrics.SensitivityAtSpecificity(sensitivityAtSpecificityThreshold, name='SensitivityAtSpecificity'),
                       tf.keras.metrics.FalsePositives(name='fp'),
                       tf.keras.metrics.FalseNegatives(name='fn'),
                       tf.keras.metrics.TruePositives(name='tp'),
                       tf.keras.metrics.TrueNegatives(name='tn')
                   ])

model.summary()

def train(verbose=True):
    #                           class_weight={
    #                            True: num_sequences / num_positives,
    #                            False: num_sequences / num_negatives
    #                          } if not multi_label else None,

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    epochs=200
    patience=4
    lr_patience=2

    checkpoint = ModelCheckpoint('data/model/gtex', monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min', save_freq='epoch')
    earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=patience) # minimize loss;
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=lr_patience) # xxx prolly not necessary for overtraining, 
    # where lr needs to be small anyway because converges quickly, but if so start learning at 0.01

    train_data=tf.data.Dataset.from_tensor_slices((x_validation,y_validation))
    train_data=train_data.repeat().shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    #history = model.fit(train_data, epochs=15, steps_per_epoch=100)
    return model.fit(x_train, y_train, 
        steps_per_epoch=y_train.shape[0]/batch_size,
        batch_size=batch_size, 
        epochs=epochs, 
        initial_epoch= 0, 
        validation_data=(x_validation, y_validation),
        callbacks=[earlystop, reduce, checkpoint],
        class_weight = d_class_weights,
        verbose=verbose
        )

def compare_predictions(x_test, y_test, y_pred):
    '''
    returns pairs of (<truth><false-prediction>) tissue names
    '''
    predicted=np.argmax(y_pred, axis=1)
    observed=np.argmax(y_test,axis=1)
    pairs=list(zip(class_names[observed[(predicted != observed)]],class_names[predicted[(predicted != observed)]]))
    return pairs


####
# TRAIN, EVALUATE, and SAVE
####
history=train(verbose=0)

# check it trained ok
print(f"Performance: ")
performance = model.evaluate(x_test,y_test)
print("Performance details: ")
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

# check predictions
print("")
y_pred = model.predict(x_test)
print(f"Number of training samples: {len(y_train)}")
print(f"Number of validation samples: {len(y_validation)}")
print(f"Number of test samples: {len(y_test)}")
print("Mis-classifications:")
print("(<truth>,<false-prediction>)")
print(compare_predictions(x_test, y_test, y_pred))

# plot and save the loss curve
import os
os.makedirs("data/images")
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.savefig('data/images/nn-loss-curve.png')

# save and make sure it saved OK
model.save('data/model/gtex/manual/gtex_model.h5')
from keras.models import load_model
savedModel=load_model('data/model/gtex/manual/gtex_model.h5')
print("")
print("Saved model summary:")
savedModel.summary()





