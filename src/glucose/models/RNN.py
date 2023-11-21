"""src/glucose/models/RNN.py

This RNN script file contains the RNN implementations
of our model.

Classes
-------
    BasicRNN
        a simple RNN implementation
"""

from .model import BaseModel
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Optimizer, Adam, RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, SimpleRNN, Bidirectional, Dense, Input, Dropout


class BasicRNN(BaseModel):
    """SimpleRNN implementation
    """

    def __init__(self, scaler, input_length, data_type):
        self.pred_horizon = None
        self.model = None
        self.len_input = input_length
        self.data_type = data_type
        self.scaler = scaler

        # init model class
        super().__init__()

    def init_network(self, pred_horizons: dict):
        """

        Parameters
        ----------
        pred_horizons: dict
            should be a dict containing the name + prediction horizon in units, e.g.
            {'15min' : 3} (in case of GCM prediction)

        Returns
        -------
        Model
            returns a Keras model which is initialized

        """

        self.pred_horizon = pred_horizons

        # input layer
        input1 = Input(batch_shape=(None, self.len_input, 1))

        # first layer with 32 neurons
        lay1 = Bidirectional(SimpleRNN(32,
                                       activation='relu'),
                             input_shape=(self.len_input, 1))(input1)

        lay2 = Dense(16)(lay1)

        # Dropout
        lay3 = Dropout(0.1)(lay2)

        # pred horizons
        outs = [Dense(1, name=x)(lay3) for x in self.pred_horizon.keys()]

        self.model = Model(inputs=input1, outputs=outs)

        return self.model

    def compile(self, optimizer=Adam(lr=0.01, decay=0.001), *args, **kwargs):
        # Create optimizer object and compile the model
        self.model.compile(optimizer=optimizer,
                           *args,
                           **kwargs)

    def fit(self, x, y, *args, **kwargs):
        self.model.fit(x, y, *args, **kwargs)
