import abc
import enum
from typing import Any

import torch.nn as nn


class ModelMode(enum.Enum):
    TRAIN = 0
    TEST = 1


class BaseModel(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, mode: ModelMode = ModelMode.TEST):
        super().__init__()
        self._mode = mode

    def set_mode(self, value: ModelMode):
        self._mode = value

    @abc.abstractmethod
    def _preprocess(self, x: Any):
        pass

    @abc.abstractmethod
    def _forward_test(self, x: Any):
        pass

    @abc.abstractmethod
    def _forward_train(self, x: Any):
        pass

    def forward(self, x: Any):
        x = self._preprocess(x)
        if self._mode == ModelMode.TEST:
            x = self._forward_test(x)
        elif self._mode == ModelMode.TRAIN:
            x = self._forward_train(x)
        return x
