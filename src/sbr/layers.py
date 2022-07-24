import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout

from keras import activations

# suppress this warning:
# WARNING:absl:Found untraced functions such as ...
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

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
                 # need to rename activation and set it to 'None' so Dense doesn't use it, 
                 # but still want user to be able to use reactivation, 
                 # but need an argument to __init__ so get_config can use and save it in the model.save function
                 # so...
                 # make another argument 'proxy_activation' and hope the user doesn't call it, 
                 # then override 'proxy_activation' value with the value of the 'activation' argument above
                 # that way you can keep the self.activation = None, but you still have proxy_activation for get_config
                 # There's got to be a better way this is an ugly kludge!
                 proxy_activation = None, 
                 dropout_rate = 0.50, 
                 trainable= True,
                 **kwargs
                 ):
        super(BADBlock, self).__init__(units, activation=None, **kwargs)
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        self.proxy_activation = activation

    def build(self, input_shape):
        super(BADBlock, self).build(input_shape)
        self.batch_layer = BatchNormalization(name = "hidden_batch_normalization")
        self.activation_layer = Activation(self.proxy_activation, name = "hidden_activation")
        self.dropout_layer = Dropout(self.dropout_rate, name = 'hidden_dropout')
              
    def call(self, inputs):
        x = self.batch_layer(super(BADBlock, self).call(inputs))
        x = self.activation_layer(x)
        return self.dropout_layer(x)

    def get_config(self):
        config = super(BADBlock,self).get_config()
        config["dropout_rate"] = self.dropout_rate
        config["trainable"] = self.trainable
        config["proxy_activation"] = self.proxy_activation

        return config
