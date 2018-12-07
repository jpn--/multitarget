
import pandas
import numpy
import warnings
import scipy.stats
from sklearn.linear_model import LinearRegression as _sklearn_LinearRegression


class LinearRegression(_sklearn_LinearRegression):

	def fit(self, X, y, sample_weight=None):
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

		return self

	def predict(self, X):
		return super().predict(X)

