
import pandas
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import make_pipeline
from .cross_val import CrossValMixin
from .base import MultiOutputRegressor
from .select import SelectNAndKBest, feature_concat
from . import ignore_warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import cross_val_score, cross_val_predict
from .detrend import DetrendMixin

class StackedSingleTargetRegression(
		BaseEstimator,
		RegressorMixin,
		CrossValMixin,
):

	def __init__(
			self,
			keep_other_features=3,
			step2_cv_folds=5,
	):
		"""

		Parameters
		----------
		keep_other_features : int
			The number of other (derived) feature columns to keep. Keeping this
			number small help prevent overfitting problems if the number of
			output features is large.
		step2_cv_folds : int
			The step 1 cross validation predictions are used in step two.  How many
			CV folds?
		"""

		self.keep_other_features = keep_other_features
		self.step2_cv_folds = step2_cv_folds


	def fit(self, X, Y):
		"""
		Fit linear and gaussian model.

		Parameters
		----------
		X : numpy array or sparse matrix of shape [n_samples, n_features]
			Training data
		T : numpy array of shape [n_samples, n_targets]
			Target values.

		Returns
		-------
		self : returns an instance of self.
		"""

		with ignore_warnings(DataConversionWarning):

			self.step1 = MultiOutputRegressor(GaussianProcessRegressor())
			Y_cv = cross_val_predict(self.step1, X, Y, cv=self.step2_cv_folds)
			self.step1.fit(X, Y)


			self.step2 = MultiOutputRegressor(
				make_pipeline(
					SelectNAndKBest(n=X.shape[1], k=self.keep_other_features),
					GaussianProcessRegressor(),
				)
			)

			self.step2.fit(feature_concat(X, Y_cv), Y)

			if isinstance(Y, pandas.DataFrame):
				self.Y_columns = Y.columns
			elif isinstance(Y, pandas.Series):
				self.Y_columns = Y.name
			else:
				self.Y_columns = None

		return self

	def predict(self, X, return_std=False, return_cov=False):
		"""Predict using the model

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = (n_samples, n_features)
			Samples.
		return_std, return_cov : bool
			Not implemented.

		Returns
		-------
		C : array, shape = (n_samples,)
			Returns predicted values.
		"""

		Yhat1 = self.step1.predict(X)
		Yhat2 = self.step2.predict(feature_concat(X, Yhat1))
		return Yhat2



class DetrendedStackedSingleTargetRegression(
	StackedSingleTargetRegression,
	DetrendMixin
):

	def fit(self, X, Y):
		return super().fit(X, self.detrend_fit(X,Y))

	def predict(self, X, return_std=False, return_cov=False):
		return self.detrend_predict(X) + super().predict(X)
