

import pandas
import numpy
import warnings

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression

from . import ignore_warnings
from .linear import LinearRegression
from .cross_val import CrossValMixin
from .detrend import DetrendMixin
from .base import MultiOutputRegressor
from .select import SelectNAndKBest, feature_concat

def _make_as_vector(y):
	# if isinstance(y, (pandas.DataFrame, pandas.Series)):
	# 	y = y.values.ravel()
	return y



class MultipleTargetRegression(
		BaseEstimator,
		RegressorMixin,
		CrossValMixin,
):

	def __init__(
			self,
			standardize_before_fit=True,
	):
		self.standardize_before_fit = standardize_before_fit
		if standardize_before_fit:
			self._kernel_generator = lambda dims: RBF([1.0] * dims)
		else:
			self._kernel_generator = lambda dims: C() * RBF([1.0] * dims)

	def fit(self, X, Y):
		"""
		Fit linear and gaussian model.

		Parameters
		----------
		X : array-like or sparse matrix of shape [n_samples, n_features]
			Training data
		Y : array-like of shape [n_samples, n_targets]
			Target values.

		Returns
		-------
		self : returns an instance of self.
		"""

		with ignore_warnings(DataConversionWarning):

			self.step1 = MultiOutputRegressor(GaussianProcessRegressor(
				kernel=self._kernel_generator(X.shape[1])
			))

			if self.standardize_before_fit:
				self.standardize_Y = Y.std(axis=0, ddof=0)
				Y = Y / self.standardize_Y
			else:
				self.standardize_Y = None

			self.step1.fit(X, Y)

			if isinstance(Y, pandas.DataFrame):
				self.Y_columns = Y.columns
			elif isinstance(Y, pandas.Series):
				self.Y_columns = Y.name
			else:
				self.Y_columns = None

		return self

	def predict(self, X, return_std=False, return_cov=False):
		"""Predict using the model

		This function will return a pandas DataFrame instead of
		a simple numpy array if there is information available
		to populate the index (if the X argument to this function
		is a DataFrame) or the columns (if the Y argument to `fit`
		was a DataFrame).

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = (n_samples, n_features)
			Samples.
		return_std, return_cov : bool
			Not implemented.

		Returns
		-------
		C : array-like, shape = (n_samples, n_targets)
			Returns predicted values. The n_targets dimension is
			determined in the `fit` merthod.
		"""

		if return_std or return_cov:
			raise NotImplementedError('return_std' if return_std else 'return_cov')

		if isinstance(X, pandas.DataFrame):
			idx = X.index
		else:
			idx = None

		Yhat1 = self.step1.predict(X)

		if self.standardize_Y is not None:
			Yhat1 *= self.standardize_Y[None,:]

		cols = None
		if self.Y_columns is not None:
			if len(self.Y_columns) == Yhat1.shape[1]:
				cols = self.Y_columns

		if idx is not None or cols is not None:
			return pandas.DataFrame(
				Yhat1,
				index=idx,
				columns=cols,
			)
		return Yhat1

	def scores(self, X, Y, sample_weight=None):
		"""
		Returns the coefficients of determination R^2 of the prediction.

		The coefficient R^2 is defined as (1 - u/v), where u is the residual
		sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
		sum of squares ((y_true - y_true.mean()) ** 2).sum().
		The best possible score is 1.0 and it can be negative (because the
		model can be arbitrarily worse). A constant model that always
		predicts the expected value of y, disregarding the input features,
		would get a R^2 score of 0.0.

		Notes
		-----
		R^2 is calculated by weighting all the targets equally using
		`multioutput='raw_values'`.  See documentation for
		sklearn.metrics.r2_score for more information.

		Parameters
		----------
		X : array-like, shape = (n_samples, n_features)
			Test samples. For some estimators this may be a
			precomputed kernel matrix instead, shape = (n_samples,
			n_samples_fitted], where n_samples_fitted is the number of
			samples used in the fitting for the estimator.

		Y : array-like, shape = (n_samples, n_outputs)
			True values for X.

		sample_weight : array-like, shape = [n_samples], optional
			Sample weights.

		Returns
		-------
		score : ndarray
			R^2 of self.predict(X) wrt. Y.
		"""
		return r2_score(Y, self.predict(X), sample_weight=sample_weight,
						multioutput='raw_values')


class DetrendedMultipleTargetRegression(
	MultipleTargetRegression,
	DetrendMixin
):

	def fit(self, X, Y):
		return super().fit(X, self.detrend_fit(X,Y))

	def predict(self, X, return_std=False, return_cov=False):
		return self.detrend_predict(X) + super().predict(X)

