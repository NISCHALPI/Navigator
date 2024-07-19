"""Functions to provide residual analysis for Kalman Filters.

Available functions:

- MSE: Calculate the Mean Squared Error (MSE) of the residuals.
- MAD: Calculate the Mean Absolute Deviation (MAD) of the residuals.
- MSSE: Calculate the Mean Squared Scaled Error (MSSE) of the residuals.
- standerized_innovations: Calculate the standerized innovation using its covariance matrix.

"""

from .kf_residuals import MAD, MSE, MSSE, standerized_innovations
