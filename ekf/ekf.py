import sys
import math
import time
import collections.abc
import numpy as np
import sympy as sp

class CoordFrame(collections.abc.Sequence):
  def __init__(self, labels):
    self._data = list(labels)
  def __len__(self):
    return len(self._data)
  def __getitem__(self, key):
    return self._data[key]
  def slice_arg(self, sarg):
    return self.index(sarg) if sarg is not None else None
  def itemkey(self, arg):
    if isinstance(arg, slice): return slice(self.slice_arg(arg.start), self.slice_arg(arg.stop), arg.step)
    elif isinstance(arg, list): return [self.index(i) for i in arg]
    else: return self.index(arg)
  def make(self, entries):
    return sp.Matrix(len(self._data), 1, entries)
  def make_zeros(self):
    return sp.Matrix.zeros(len(self._data), 1)
  def make_ones(self):
    return sp.Matrix.ones(len(self._data), 1)

class MatrixFrame:
  def __init__(self, rfrm, cfrm):
    self.rfrm = rfrm
    self.cfrm = cfrm
  def make(self, entries):
    if isinstance(entries, np.ndarray): return sp.Matrix(entries)
    else: return sp.Matrix(len(self.rfrm), len(self.cfrm), entries)
  def make_zeros(self):
    return sp.Matrix.zeros(len(self.rfrm), len(self.cfrm))
  def make_ones(self):
    return sp.Matrix.ones(len(self.rfrm), len(self.cfrm))
  def make_diag(self, entries):
    if len(self.rfrm) != len(self.cfrm):
      raise Exception("Matrix is not a square type: (%d, %d)" % (len(self.rfrm), len(self.cfrm)))
    if len(entries) != len(self.rfrm):
      raise Exception("Invalid number of entries: Expected %d, got %d" % (len(self.rfrm), len(entries)))
    return sp.Matrix.diag(entries)
  def make_eye(self):
    if len(self.rfrm) != len(self.cfrm):
      raise Exception("Matrix is not a square type: (%d, %d)" % (len(self.rfrm), len(self.cfrm)))
    return sp.Matrix.eye(len(self.rfrm))

class CoordVectorBase(object):
  def __init__(self, frame, matrix):
    self.frame = frame
    self.matrix = matrix
  def __getitem__(self, arg):
    return self.matrix[self.frame.itemkey(arg),0]
  def __setitem__(self, arg, value):
    self.matrix[self.frame.itemkey(arg),0] = value
class CoordVector(CoordVectorBase):
  def __init__(self, frame, entries):
    super().__init__(frame, sp.Matrix(len(frame), 1, entries))        
class CoordVectorSymbol(CoordVectorBase):
  def __init__(self, name, frame):
    super().__init__(frame, sp.MatrixSymbol(name, len(frame), 1))
    self.name = name
class CoordVectorFunction(CoordVectorBase):
  def __init__(self, frame, argsyms, entries):
    self.argsyms = argsyms
    super().__init__(frame, entries if isinstance(entries, sp.Matrix) else sp.Matrix(len(frame), 1, entries))
  def evalf(self, *args, **kwargs):
    if len(args) != len(self.argsyms):
      raise ValueError("Incorrect number of arguments. Expexted %d, got %d" % (len(self.argsyms), len(args)))
    subs = [(argsym.matrix, sp.Matrix(args[i])) for i, argsym in enumerate(self.argsyms)]
    return self.matrix.subs(subs).evalf()
  def jacobian(self, sym):
    syms = [sym] if (not isinstance(sym, tuple) and not isinstance(sym, list)) else sym
    return CoordVectorFunction(self.frame, self.argsyms, self.matrix.jacobian([s.matrix for s in syms]))


class EKFDesc(object):
  __slots__ = ['x', 'u', 'z', 'f', 'h', 'c', 'xf', 'uf', 'zf', 'cf', 'dfdx', 'dhdx', 'Pf', 'Qf', 'Rf']
  def __init__(self, x, u, z, f, h, c):
    self.xf = x.frame
    self.uf = u.frame
    self.zf = z.frame
    self.cf = c.frame

    self.x = x.matrix
    self.u = u.matrix
    self.z = z.matrix
    self.c = c.matrix
        
    self.f = f
    self.dfdx = self.f.jacobian(x)
    self.h = h
    self.dhdx = self.h.jacobian(x)

    self.Pf = MatrixFrame(self.xf, self.xf)
    self.Qf = MatrixFrame(self.xf, self.xf)
    self.Rf = MatrixFrame(self.zf, self.zf)


class EKF:
  def __init__(self, desc):
    self.desc = desc
    self.k = 0
    self.x_k = self.desc.xf.make_zeros()
    self.P_k = self.desc.Pf.make_zeros()
    self.Q_0 = self.desc.Qf.make_zeros()
    self.R_0 = self.desc.Rf.make_zeros()
    self.c_k = self.desc.cf.make_zeros()

  def predict(self, u_k, Q_k=None):
    self.k += 1
    self.u_k = self.desc.uf.make(u_k)
    self.Q_k = self.desc.Qf.make(Q_k) if Q_k is not None else self.Q_0

    self.f_k = self.desc.f.evalf(self.x_k, self.u_k, self.c_k)
    self.F_k = self.desc.dfdx.evalf(self.x_k, self.u_k, self.c_k)
        
    self.x_pred_k = self.f_k
    self.P_pred_k = (self.F_k * self.P_k * self.F_k.transpose()) + self.Q_k

  def update(self, z_k, R_k=None):
    self.z_k = self.desc.zf.make(z_k)
    self.R_k = self.desc.Rf.make(R_k) if R_k is not None else self.R_0

    self.h_k = self.desc.h.evalf(self.x_pred_k, self.c_k)
    self.H_k = self.desc.dhdx.evalf(self.x_pred_k, self.c_k)

    self.y_k = self.z_k - self.h_k
    self.S_k = (self.H_k * self.P_pred_k * self.H_k.transpose()) + self.R_k
    self.K_k = self.P_pred_k * self.H_k.transpose() * self.S_k.inv()

    self.x_est_k = self.x_pred_k + self.K_k * self.y_k
    self.P_est_k = (self.desc.Pf.make_eye() - self.K_k * self.H_k) * self.P_pred_k

    self.x_k = self.x_est_k.evalf()
    self.P_k = self.P_est_k.evalf()




