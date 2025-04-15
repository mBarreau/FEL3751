# Installation

Create a virtual environment running on Python 3.12. Install the requirements:
```
pip install -r requirements.txt
```

# Run the code

## Classic behavior

You just have to run the `main.py` file. It is possible to use VScode jupyter notebook instead.
The control signal must be a column vector. Multiple columns indicate different time.

## Test new things

We can change the behavior of the sensor. To add measurement noise, change the `noise_level` variable. To change the dynamics of the sensor, please consider the following scenarios:
1) If we want `y(t) = h(x(t))` then setting the variable `tau = 0` is sufficient.
2) If we want `y(t) = h(x(t-tau))` then the variable `tau` must be strictly positive.
3) In case of a moving average `y(t) = \int_{t-tau}^t h(x(s)) ds` then the variable `tau` must be strictly positive and `method = "moving_average"`. 

# Future perspectives

1) Test and write the outcomes with the current code
2) We can investigate the observability conditions on non linear systems
3) We can compute the control action `u` based on some state `y`