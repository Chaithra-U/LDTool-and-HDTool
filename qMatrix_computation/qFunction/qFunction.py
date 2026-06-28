import pandas as pd
import numpy as np

def q_functionDf(df: pd.DataFrame, col_A_name: str, col_B_name: str) -> float:
    """
    Calculates the Q-score for two columns, measuring the logical dependency
    of col_B on col_A. A score of 0 indicates perfect functional dependency (A -> B).
    """
    if col_A_name not in df.columns or col_B_name not in df.columns:
        raise ValueError("One or both column names are not in the DataFrame.")

    unique_A = df[col_A_name].dropna().unique()
    unique_B = df[col_B_name].dropna().unique()

    num_unique_A = len(unique_A)
    num_unique_B = len(unique_B)

    if num_unique_A == 0 or num_unique_B <= 1:
        return 0.0

    num_observed_pairs = df[[col_A_name, col_B_name]].dropna().drop_duplicates().shape[0]

    numerator = (num_observed_pairs / num_unique_A) - 1
    denominator = num_unique_B - 1

    if denominator == 0:
        return 0.0

    return numerator / denominator


def q_functionNp(df: np.ndarray, col_A: int, col_B: int) -> float:
    """
    Calculates the Q-score for two columns, measuring the logical dependency
    of col_B on col_A. A score of 0 indicates perfect functional dependency (A -> B).
    """
    assert( isinstance(df, np.ndarray) )
    assert( len(df.shape) == 2 )
    
    if min(df.shape[0], df.shape[1]) < 1:
        raise ValueError("Array has to contain at least one element.")

    if min(col_A, col_B < 0) or max(col_A, col_B) >= df.shape[1]:
        raise ValueError("One or both column index are out of range.")

    if col_A == col_B:
      return 0.0

    num_unique_A = len(set(df[:,col_A]))
    num_unique_B = len(set(df[:,col_B]))

    if num_unique_A == 0 or num_unique_B <= 1:
        return 0.0

    num_observed_pairs = len(set(zip(df[:,col_A], df[:,col_B]))) 

    numerator = (num_observed_pairs / num_unique_A) - 1
    denominator = num_unique_B - 1

    if denominator == 0:
        return 0.0

    return numerator / denominator


def qMatrix(df: np.ndarray):
  qf = QFunction(df)
  return qf.qValues, qf



class QBase:
  def __init__(self, data):
    assert( isinstance(data, np.ndarray) )
    assert( len(data.shape) == 2 )
    assert( data.shape[0] >= 1 )
    assert( data.shape[1] >= 1 )
    self.data = data
    self.nFeatures = data.shape[1]
    self.nRows = data.shape[0]

    self.qValues = np.zeros(shape=(self.nFeatures, self.nFeatures), dtype=float)
    self.setSizes = np.zeros(shape=(self.nFeatures, ), dtype=float) - 1
    self.pairSizes = np.zeros(shape=(self.nFeatures, self.nFeatures), dtype=float) - 1

    self.nRequested = 0
    self.nSetRequested = 0
    self.nPairRequested = 0
    self.nCalculated = 0
    self.nSetCalculated = 0
    self.nPairCalculated = 0

    for i in range(self.nFeatures):
      for j in range(self.nFeatures):
        self.qValues[i,j] = None


  def setSize(self, n, useBuffer=True):
    self.nSetRequested += 1
    if n < 0 or n >= self.nFeatures:
      return 0

    if useBuffer:
      if self.setSizes[n] < 0:
        self.nSetCalculated += 1
        self.setSizes[n] = len(set(self.data[:,n]))
      return self.setSizes[n]

    self.nSetCalculated += 1
    return len(set(self.data[:,n]))

  def pairSize(self, i, j, useBuffer=True):
    self.nPairRequested += 1
    if min(i, j) < 0 or max(i,j) >= self.nFeatures:
      return 0

    if useBuffer:
      if self.pairSizes[i,j] < 0:
        self.nPairCalculated += 1
        s = len(set(zip(self.data[:, i], self.data[:, j])))
        self.pairSizes[i,j] = s
        self.pairSizes[j,i] = s

      return self.pairSizes[i,j]

    self.nPairCalculated += 1
    return len(set(zip(self.data[:, i], self.data[:, j])))

  def statistics(self):
    print(f"nRequested = {self.nRequested}")
    print(f"nSetRequested = {self.nSetRequested}")
    print(f"nPairRequested = {self.nPairRequested}")
    print(f"nCalculated = {self.nCalculated}")
    print(f"nSetCalculated = {self.nSetCalculated}")
    print(f"nPairCalculated = {self.nPairCalculated}")

    print(f"Optimized Q-matrix would be:")
    print(f"nCalculated = {(self.nFeatures * (self.nFeatures - 1)) / 2}")
    print(f"nSetCalculated = {self.nFeatures}")
    print(f"nPairCalculated = {(self.nFeatures * (self.nFeatures - 1)) / 2}")


class Q(QBase):
  def __init__(self, data):
    super().__init__(data)

  def calc(self, i, j):
    self.nRequested += 1
    q = 0.0
    if i != j:
      self.nCalculated += 1
      a = self.setSize(i, False)
      b = self.setSize(j, False)
      if a >= 1 and b >= 2:
        r = self.pairSize(i, j, False)
        q = ((r / a) - 1) / (b - 1)

    if np.isnan(self.qValues[i,j]):
      self.qValues[i,j] = q

    return q






class Qfast(QBase):
  def __init__(self, data):
    super().__init__(data)

  def calc(self, i, j):
    self.nRequested += 1
    if np.isnan(self.qValues[i, j]):
      self.nCalculated += 1
      q1 = 0.0
      q2 = 0.0
      if i != j:
        a = self.setSize(i)
        b = self.setSize(j)
        r = None
        if a >= 1 and b >= 2:
          r = self.pairSize(i,j)
          q1 = ((r / a) - 1) / (b - 1)
        if a >= 2 and b >= 1:
          if r is None:
            r = self.pairSize(i,j)
          q2 = ((r / b) - 1) / (a - 1)
      self.qValues[i, j] = q1
      self.qValues[j, i] = q2
    return self.qValues[i, j]




class QFunction(QBase):
  def __init__(self, data):
    super().__init__(data)

    def q(i,j):
      si = self.setSize(i)
      sj = self.setSize(j)

      if i == j or si < 1 or sj < 1:
        return 0.0, 0.0

      if si == self.nRows and sj == self.nRows:
        return 0.0, 0.0

      self.nCalculated += 1
      nPairs = self.pairSize(i, j)
      q1, q2 = 0.0, 0.0
      if sj >= 2 and si < self.nRows:
        q1 = ((nPairs / si) - 1) / (sj - 1)

      if si >= 2 and sj < self.nRows:
        q2 = ((nPairs / sj) - 1) / (si - 1)
      return q1, q2

    for i in range(self.nFeatures):
      for j in range(self.nFeatures):
        q1, q2 = q(i, j)
        self.qValues[i,j] = q1
        self.qValues[j,i] = q2

  def calc(self, i, j):
    self.nRequested += 1
    return self.qValues[i,j]





