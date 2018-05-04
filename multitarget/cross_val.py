

import pandas, numpy
from . import ignore_warnings

from sklearn.metrics import r2_score
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import cross_val_score, cross_val_predict


class CrossValMixin:

	def cross_val_scores(self, X, Y, cv=3):
		p = self.cross_val_predict(X, Y, cv=cv)
		return pandas.Series(
			r2_score(Y, p, sample_weight=None, multioutput='raw_values'),
			index=Y.columns
		)

	def cross_val_predict(self, X, Y, cv=3):
		if isinstance(Y, pandas.DataFrame):
			self.Y_columns = Y.columns
			Yix = Y.index
		elif isinstance(Y, pandas.Series):
			self.Y_columns = [Y.name]
			Yix = Y.index
		else:
			self.Y_columns = ["Untitled" * Y.shape[1]]
			Yix = pandas.RangeIndex(Y.shape[0])
		with ignore_warnings(DataConversionWarning):
			p = cross_val_predict(self, X, Y, cv=cv)
		return pandas.DataFrame(p, columns=self.Y_columns, index=Yix)
