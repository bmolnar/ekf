import sys
import math
import time
import collections
import numpy as np
import sympy as sp


class CoordFrame(collections.UserList):
  def index_or_slice(self, arg):
    if isinstance(arg, slice):
      lbltoidx = lambda x: self.index(x) if x is not None else None
      return slice(lbltoidx(arg.start), lbltoidx(arg.stop), arg.step)
    else:
      return self.index(arg)
class CoordVectorBase(object):
  def __init__(self, frame, matrix):
    self.frame = frame
    self.matrix = matrix
  def __getitem__(self, arg):
    index = self.frame.index_or_slice(arg)
    return self.matrix[index,0]
  def __setitem__(self, arg, value):
    index = self.frame.index_or_slice(arg)
    self.matrix[index,0] = value
class CoordVector(CoordVectorBase):
  def __init__(self, frame, entries):
    super().__init__(frame, sp.Matrix(len(frame), 1, entries))        
class CoordVectorSymbol(CoordVectorBase):
  def __init__(self, name, frame):
    super().__init__(frame, sp.MatrixSymbol(name, len(frame), 1))
    self.name = name
    #self.comp = {lbl: self.mtx[idx,0] for idx, lbl in enumerate(self.frm.labels)}



class EKFDesc(object):
  __slots__ = ['x', 'u', 'z', 'f', 'h', 'c', 'xf', 'uf', 'zf', 'cf', 'dfdx', 'dhdx']
  def __init__(self, x, u, z, f, h, c):
    self.x = x.matrix
    self.u = u.matrix
    self.z = z.matrix
    self.f = f.matrix
    self.h = h.matrix
    self.c = c.matrix
        
    self.xf = x.frame
    self.uf = u.frame
    self.zf = z.frame
    self.cf = c.frame

    self.dfdx = self.f.jacobian(self.x)
    self.dhdx = self.h.jacobian(self.x)

  def eval_f(self, x_values, u_values, c_values):
    subs = [(self.x, x_values), (self.u, u_values), (self.c, c_values)]
    return self.f.subs(subs).evalf()
  def eval_dfdx(self, x_values, u_values, c_values):
    subs = [(self.x, x_values), (self.u, u_values), (self.c, c_values)]
    return self.dfdx.subs(subs).evalf()
  def eval_h(self, x_values, c_values):
    subs = [(self.x, x_values), (self.c, c_values)]
    return self.h.subs(subs).evalf()
  def eval_dhdx(self, x_values, c_values):
    subs = [(self.x, x_values), (self.c, c_values)]
    return self.dhdx.subs(subs).evalf()

  def make_x(self, entries):
    return sp.Matrix(self.x.rows, self.x.cols, entries)
  def make_u(self, entries):
    return sp.Matrix(self.u.rows, self.u.cols, entries)
  def make_z(self, entries):
    return sp.Matrix(self.z.rows, self.z.cols, entries)
  def make_c(self, entries):
    return sp.Matrix(self.c.rows, self.c.cols, entries)
 
  def make_Q(self, entries):
    return sp.Matrix(self.x.rows, self.x.rows, entries)
  def make_R(self, entries):
    return sp.Matrix(self.z.rows, self.z.rows, entries)

  def make_diag_Q(self, entries):
    if len(entries) != self.x.rows:
      raise Exception("Invalid number of components. Exepected %d, got %d" % (self.x.rows, len(entries)))
    return sp.Matrix.diag(entries)
  def make_diag_R(self, entries):
    if len(entries) != self.z.rows:
      raise Exception("Invalid number of components. Exepected %d, got %d" % (self.z.rows, len(entries)))
    return sp.Matrix.diag(entries) 
    

class EKF:
  def __init__(self, desc):
    self.desc = desc

    self.k = 0
    self.x_k = sp.Matrix.zeros(self.desc.x.rows, self.desc.x.cols)
    self.P_k = sp.Matrix.zeros(self.desc.x.rows, self.desc.x.rows)
    self.Q_k = sp.Matrix.zeros(self.desc.x.rows, self.desc.x.rows)
    self.R_k = sp.Matrix.zeros(self.desc.z.rows, self.desc.z.rows)
    self.c_k = sp.Matrix.zeros(self.desc.c.rows, self.desc.c.cols)


  def predict(self, u_k, Q_k=None):
    self.k += 1

    self.u_k = self.desc.make_u(u_k)
    self.Q_k = self.desc.make_Q(Q_k) if Q_k is not None else self.Q_k

    #self.subs_pred = [(self.desc.c, self.c_k), (self.desc.x, self.x_k), (self.desc.u, self.u_k)]
    #self.f_k = self.desc.f.subs(self.subs_pred).evalf()
    #self.F_k = self.desc.dfdx.subs(self.subs_pred).evalf()
    self.f_k = self.desc.eval_f(self.x_k, self.u_k, self.c_k)
    self.F_k = self.desc.eval_dfdx(self.x_k, self.u_k, self.c_k)
        
    self.x_pred_k = self.f_k
    self.P_pred_k = (self.F_k * self.P_k * self.F_k.transpose() + self.Q_k)

  def update(self, z_k, R_k=None):
    self.z_k = self.desc.make_z(z_k)
    self.R_k = self.desc.make_R(R_k) if R_k is not None else self.R_k

    #self.subs_update = [(self.desc.c, self.c_k), (self.desc.x, self.x_pred_k)]
    #self.h_k = self.desc.h.subs(self.subs_update).evalf()
    #self.H_k = self.desc.dhdx.subs(self.subs_update).evalf()
    self.h_k = self.desc.eval_h(self.x_pred_k, self.c_k)
    self.H_k = self.desc.eval_dhdx(self.x_pred_k, self.c_k)

    self.y_k = self.z_k - self.h_k
    self.S_k = self.H_k * self.P_pred_k * self.H_k.transpose() + self.R_k
    self.K_k = self.P_pred_k * self.H_k.transpose() * self.S_k.inv()

    self.x_est_k = self.x_pred_k + self.K_k * self.y_k
    self.P_est_k = (sp.Matrix.eye(self.desc.x.rows) - self.K_k * self.H_k) * self.P_pred_k

    self.x_k = self.x_est_k.evalf()
    self.P_k = self.P_est_k.evalf()




