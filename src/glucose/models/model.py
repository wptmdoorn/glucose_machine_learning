"""src/glucose/models/models.py

This models script contains the base models class
to be used when creating a new models. It should be
capable of initializing, saving, validation the models.
Also we should have a save/load method to conveniently save
these algorithms.

Classes
-------
    Model
        base class of algorithms which should be used when a new models
        is implemented.
"""

from enum import Enum
from keras.models import Sequential


class DATATYPE(Enum):
    GCM = 1
    GCMACTIVPAL = 2


class BaseModel(object):
    """Model base class for implementing a model.

    A new instance of this class should implement the functions:
        - init_network()
    Please refer to the manual for the exact documentation.
    """

    def _init__(self, input_length: int, data_type: DATATYPE):
        """Initialize base class

        Parameters
        ----------
        input_length: int
            input length of the data
        data_type
            data type which is used according to model.DATATYPE
        """

        self.len_input = input_length
        self.data_type = data_type

    def init_network(self, *args, **kwargs):
        raise NotImplementedError('Subclass must override init_network()!')

    def compile(self, *args, **kwargs):
        raise NotImplementedError('Subclass must override compile()!')

    def fit(self, *args, **kwargs):
        raise NotImplementedError('Subclass must override fit()!')

    def get_model(self):
        return self.model

    def predict(self, data):
        predicted = self.model.predict(data)
        return [self.scaler.inverse_transform(y) for y in predicted]
