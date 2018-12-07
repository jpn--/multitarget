

import pandas
import numpy
import scipy.stats
import warnings

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import make_pipeline
from .cross_val import CrossValMixin
from .base import MultiOutputRegressor
from .select import SelectNAndKBest, feature_concat
from . import ignore_warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
from .detrend import DetrendMixin
from sklearn.linear_model import LinearRegression as _sklearn_LinearRegression
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression


def _make_as_vector(y):
	# if isinstance(y, (pandas.DataFrame, pandas.Series)):
	# 	y = y.values.ravel()
	return y


class LinearRegression(_sklearn_LinearRegression):

	def fit(self, X, y, sample_weight=None):
		# print(" LR FIT on",len(X))
		super().fit(X, y, sample_weight=sample_weight)

		if isinstance(X, pandas.DataFrame):
			self.names_ = X.columns.copy()

		sse = numpy.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])

		if sse.shape == ():
			sse = sse.reshape(1,)

		inv_X_XT = numpy.linalg.inv(numpy.dot(X.T, X))

		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)

			try:
				se = numpy.array([
					numpy.sqrt(numpy.diagonal(sse[i] * inv_X_XT))
					for i in range(sse.shape[0])
				])
			except:
				print("sse.shape",sse.shape)
				print(sse)
				raise

			self.t_ = self.coef_ / se
			self.p_ = 2 * (1 - scipy.stats.t.cdf(numpy.abs(self.t_), y.shape[0] - X.shape[1]))

		# try:
		# 	print(y.values[0])
		# except AttributeError:
		# 	print(y[0])
		return self

	def predict(self, X):
		# print(" "*55,"LR PREDICT on", len(X))
		return super().predict(X)



class LinearAndGaussianProcessRegression(
		BaseEstimator,
		RegressorMixin,
):

	def __init__(self, core_features=None, keep_other_features=3, use_linear=True):
		"""

		Parameters
		----------
		core_features
			feature columns to definitely keep for both LR and GPR

		"""

		self.core_features = core_features
		self.keep_other_features = keep_other_features
		self.lr = LinearRegression()
		self.gpr = GaussianProcessRegressor(n_restarts_optimizer=9)
		self.y_residual = None
		self.kernel_generator = lambda dims: C() * RBF([1.0] * dims)
		self.use_linear = use_linear


	def _feature_selection(self, X, y=None):
		"""

		Parameters
		----------
		X : pandas.DataFrame
		y : ndarray
			If given, the SelectKBest feature selector will be re-fit to find the best features. If not given,
			then the previously fit SelectKBest will be used; if it has never been fit, an error is raised.

		Returns
		-------
		pandas.DataFrame
			Contains all the core features plus the K best other features.
		"""

		if not isinstance(X, pandas.DataFrame):
			raise TypeError('must use pandas.DataFrame for X')

		if self.core_features is None:
			return X

		y = _make_as_vector(y)
		X_core = X.loc[:,self.core_features]
		X_other = X.loc[:, X.columns.difference(self.core_features)]
		if X_other.shape[1] <= self.keep_other_features:
			return X

		# If self.keep_other_features is zero, there is no feature selecting to do and we return only the core.
		if self.keep_other_features == 0:
			return X_core

		if y is not None:
			self.feature_selector = SelectKBest(mutual_info_regression, k=self.keep_other_features).fit(X_other, y)

		try:
			X_other = pandas.DataFrame(
				self.feature_selector.transform(X_other),
				columns=X_other.columns[self.feature_selector.get_support()],
				index=X_other.index,
			)
		except:
			print("X_other.info")
			print(X_other.info(1))
			print("X_other")
			print(X_other)
			raise

		try:
			return pandas.concat([X_core, X_other], axis=1)
		except:
			print("X_core")
			print(X_core)
			print("X_other")
			print(X_other)
			print(X_core.info())
			print(X_other.info())
			raise


	def fit(self, X, y):
		"""
		Fit linear and gaussian model.

		Parameters
		----------
		X : numpy array or sparse matrix of shape [n_samples, n_features]
			Training data
		y : numpy array of shape [n_samples, n_targets]
			Target values.

		Returns
		-------
		self : returns an instance of self.
		"""
		# print("META FIT on",len(X))

		# if not isinstance(X, pandas.DataFrame):
		# 	# X = pandas.DataFrame(X)
		# 	raise TypeError('must use pandas.DataFrame for X')
		#
		# if self.core_features is None:
		# 	X_core = X
		# 	X_other = X.loc[:,[]]
		# else:
		# 	X_core = X.loc[:,self.core_features]
		# 	X_other = X.loc[:, X.columns.difference(self.core_features)]
		#
		# self.feature_selector = SelectKBest(mutual_info_regression, k=self.keep_other_features).fit(X_other, y)
		#
		# X_other = self.feature_selector.transform(X_other)

		with ignore_warnings(DataConversionWarning):

			if isinstance(y, pandas.DataFrame):
				self.Y_columns = y.columns
			elif isinstance(y, pandas.Series):
				self.Y_columns = [y.name]
			else:
				self.Y_columns = None

			y = _make_as_vector(y)
			X_core_plus = self._feature_selection(X, y)

			if self.use_linear:
				try:
					self.lr.fit(X_core_plus, y)
				except:
					print("X_core_plus.shape",X_core_plus.shape)
					print("y.shape",y.shape)
					print(X_core_plus)
					print(y)
					raise
				self.y_residual = y - self.lr.predict(X_core_plus)
			else:
				self.y_residual = y
			dims = X_core_plus.shape[1]
			self.gpr.kernel = self.kernel_generator(dims)
			self.gpr.fit(X_core_plus, self.y_residual)
			# print(self.y_residual.values[0])

		return self


	def predict(self, X, return_std=False, return_cov=False):
		"""Predict using the model

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = (n_samples, n_features)
			Samples.

		Returns
		-------
		C : array, shape = (n_samples,)
			Returns predicted values.
		"""

		if not isinstance(X, pandas.DataFrame):
			raise TypeError('must use pandas.DataFrame for X')
		X_core_plus = self._feature_selection(X)

		if self.use_linear:
			y_hat_lr = self.lr.predict(X=X_core_plus)
		else:
			y_hat_lr = 0

		if return_std:
			y_hat_gpr, y_hat_std = self.gpr.predict(X_core_plus, return_std=True)

			if self.Y_columns is not None:
				y_result = pandas.DataFrame(
					y_hat_lr + y_hat_gpr,
					columns=self.Y_columns,
					index=X.index,
				)
			else:
				y_result = y_hat_lr + y_hat_gpr

			return y_result, y_hat_std
		else:
			y_hat_gpr = self.gpr.predict(X_core_plus)

			if self.Y_columns is not None:
				y_result = pandas.DataFrame(
					y_hat_lr + y_hat_gpr,
					columns=self.Y_columns,
					index=X.index,
				)
			else:
				y_result = y_hat_lr + y_hat_gpr

			return y_result

	def cross_val_scores(self, X, Y, cv=3):
		p = self.cross_val_predict(X, Y, cv=cv)
		return pandas.Series(
			r2_score(Y, p, sample_weight=None, multioutput='raw_values'),
			index=Y.columns
		)
	#
	# def cross_val_scores(self, X, y, cv=3):
	# 	with ignore_warnings(DataConversionWarning):
	# 		y = _make_as_vector(y)
	# 		X_core_plus = self._feature_selection(X, y)
	# 		total = cross_val_score(self, X_core_plus, y, cv=cv)
	# 	return total

	def cross_val_scores_full(self, X, y, cv=3, alt_y=None):

		with ignore_warnings(DataConversionWarning):
			y = _make_as_vector(y)

			X_core_plus = self._feature_selection(X, y)

			total = cross_val_score(self, X_core_plus, y, cv=cv)

			if self.use_linear:
				linear_cv_score = cross_val_score(self.lr, X_core_plus, y, cv=cv)
				linear_cv_predict = cross_val_predict(self.lr, X_core_plus, y, cv=cv)
				linear_cv_residual = y-linear_cv_predict
				gpr_cv_score = cross_val_score(self.gpr, X_core_plus, linear_cv_residual, cv=cv)

				self.lr.fit(X_core_plus, y)
				y_residual = y - self.lr.predict(X_core_plus)
				gpr_cv_score2 = cross_val_score(self.gpr, X_core_plus, y_residual, cv=cv)

				result = dict(
					total=total,
					linear=linear_cv_score,
					net_gpr=total-linear_cv_score,
					gpr=gpr_cv_score,
					gpr2=gpr_cv_score2,
				)
			else:
				result = dict(
					total=total,
				)

			if alt_y is not None:
				result['gpr_alt'] = cross_val_score(self.gpr, X, alt_y, cv=cv)
				# print()
				# print(numpy.concatenate([y_residual, alt_y, y_residual-alt_y], axis=1 ))
				# print()
				# print(result['gpr_alt'])
				# print(result['gpr2'])
				# print()
		return result

	def cross_val_predict(self, X, y, cv=3):

		with ignore_warnings(DataConversionWarning):

			X_core_plus = self._feature_selection(X, y)

			if isinstance(y, pandas.DataFrame):
				y_columns = y.columns
			elif isinstance(y, pandas.Series):
				y_columns = [y.name]
			else:
				y_columns = ['Unnamed']

			total = cross_val_predict(self, X_core_plus, y, cv=cv)
			return pandas.DataFrame(
				total,
				index=y.index,
				columns=y_columns,
			)

	def cross_val_predicts(self, X, y, cv=3):

		with ignore_warnings(DataConversionWarning):
			y = _make_as_vector(y)

			X_core_plus = self._feature_selection(X, y)

			total = cross_val_predict(self, X_core_plus, y, cv=cv)
			if self.use_linear:
				linear_cv_predict = cross_val_predict(self.lr, X_core_plus, y, cv=cv)
				linear_cv_residual = y-linear_cv_predict
				gpr_cv_predict_over_cv_linear = cross_val_predict(self.gpr, X_core_plus, linear_cv_residual, cv=cv)

				self.lr.fit(X_core_plus, y)
				linear_full_predict = self.lr.predict(X_core_plus)
				y_residual = y - linear_full_predict
				gpr_cv_predict_over_full_linear = cross_val_predict(self.gpr, X_core_plus, y_residual, cv=cv)

				return dict(
					total=total,
					linear=linear_cv_predict,
					net_gpr=total-linear_cv_predict,
					gpr=gpr_cv_predict_over_cv_linear+linear_cv_predict,
					gpr2=gpr_cv_predict_over_full_linear+linear_full_predict,
				)
			else:
				return dict(
					total=total,
				)

