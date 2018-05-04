
from sklearn.linear_model import LinearRegression

class DetrendMixin:

	def detrend_fit(self, X, Y):
		self._lr = LinearRegression()
		self._lr.fit(X, Y)
		residual = Y - self._lr.predict(X)
		return residual

	def detrend_predict(self, X):
		Yhat1 = self._lr.predict(X)
		return Yhat1
