import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout

from keras import activations

# suppress this warning:
# WARNING:absl:Found untraced functions such as ...
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

verbose = True

class BADBlock(layers.Dense):
    '''
    Dense layer followed by Batch, Activation, Dropout. When popular
    kwarg input_shape is passed, then will create a keras input layer
    to insert before the current layer to avoid explicitly defining an
    InputLayer.

    # Recreate this layer from its config:
    layer = BADBlock(1000)
    config = layer.get_config()
    print(config)
    new_layer = BADBlock.from_config(config)

    # use in a model:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    from sbr.layers import BADBlock
    model = Sequential()
    model.add(BADBlock(1000, input_dim = 18963, activation='relu', dropout_rate=0.50, name="BAD_1"))
    model.add(Dense(26, activation="softmax"))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy','mse'])
    '''
    def __init__(self, units,
                 activation = 'relu',
                 # need to rename the 'activation' parameter so the Dense superclass doesn't use it, 
                 # because the Activation layer is responsible for 'activation'.
                 # But still we want user to be able to use the 'activation' parameter.
                 # but we need an argument to __init__ so 'get_config' can use and save it in the 'model.save' function
                 # so...
                 # Make another argument, 'block_activation',
                 # then override 'block_activation' value with the value of the 'activation' argument above.
                 # That way even if 'self.activation' = 'linear', you still have 'block_activation' for 'get_config'.
                 # This is odd, but only causes strange behavior if user sets both activation and block_activation at the same time.
                 block_activation = None, 
                 dropout_rate = 0.50, 
                 trainable= True,
                 **kwargs
                 ):
        if block_activation is None:
            # this gets called when end user sets activation paramater
            self.block_activation = activation
        else:
            # this gets called when loading model from file
            self.block_activation = block_activation

        # The superclass Dense layer is just a pass through to the Activation
        super(BADBlock, self).__init__(units, activation='linear', **kwargs)

        self.dropout_rate = dropout_rate
        self.trainable = trainable


    def build(self, input_shape):
        super(BADBlock, self).build(input_shape)
        self.batch_layer = BatchNormalization(name = "hidden_batch_normalization")
        self.activation_layer = Activation(self.block_activation, name = "hidden_activation")
        self.dropout_layer = Dropout(self.dropout_rate, name = 'hidden_dropout')
              
    def call(self, inputs):
        x = self.batch_layer(super(BADBlock, self).call(inputs))
        x = self.activation_layer(x)
        return self.dropout_layer(x)

    def get_config(self):
        config = super(BADBlock,self).get_config()
        config["dropout_rate"] = self.dropout_rate
        config["trainable"] = self.trainable
        config["block_activation"] = self.block_activation

        return config
