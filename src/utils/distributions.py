# src/utils/distributions.py
import numpy as np
import random
import optuna

class CategoricalDistribution:
    def __init__(self, choices):
        if not isinstance(choices, list):
            raise TypeError("Choices must be a list.")
        if not choices:
            raise ValueError("Choices list cannot be empty.")
        self.choices = choices

    def sample(self, trial: optuna.trial.Trial, name: str):
        return trial.suggest_categorical(name, self.choices)

    def __repr__(self):
        return f"CategoricalDistribution(choices={self.choices})"

class FloatDistribution:
    def __init__(self, low, high, log=False, step=None):
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise TypeError("Low and high must be numbers.")
        if low >= high:
            raise ValueError("Low must be less than high.")
        self.low = low
        self.high = high
        self.log = log
        self.step = step

    def sample(self, trial: optuna.trial.Trial, name: str):
        return trial.suggest_float(name, self.low, self.high, log=self.log, step=self.step)

    def __repr__(self):
        return f"FloatDistribution(low={self.low}, high={self.high}, log={self.log}, step={self.step})"

class IntDistribution:
    def __init__(self, low, high, step=1, log=False):
        if not isinstance(low, int) or not isinstance(high, int):
            raise TypeError("Low and high must be integers.")
        if low > high: # Allow low == high for a fixed integer value
            raise ValueError("Low must be less than or equal to high.")
        self.low = low
        self.high = high
        self.step = step
        self.log = log # log for integers is less common but supported by Optuna

    def sample(self, trial: optuna.trial.Trial, name: str):
        return trial.suggest_int(name, self.low, self.high, step=self.step, log=self.log)

    def __repr__(self):
        return f"IntDistribution(low={self.low}, high={self.high}, step={self.step}, log={self.log})"

# It's good practice to also have an __init__.py in the utils directory,
# though it might not be strictly necessary for this specific import error
# if the parent 'src' directory is correctly on sys.path.
# Create an empty src/utils/__init__.py if it doesn't exist:
# # src/utils/__init__.py
# (This file can be empty) 