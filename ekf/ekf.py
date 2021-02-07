import sys
import math
import time
import numbers
import collections
import yaml
import sympy


g_trace = True
def trace(*args, **kwargs):
  if g_trace:
    print(*args, **kwargs)



class Dumper(yaml.Dumper):
  def ignore_aliases(self, data):
    return True





class SerializableBase(object):
  __slots__ = []


class SerializableMeta(type):
  def __new__(cls, name, bases, attr):
    #trace("SerializableMeta.__new__: cls=%s, name=%s, bases=%s, attr=%s" % (repr(cls), repr(name), repr(bases), repr(attr)))
    result = super(SerializableMeta, cls).__new__(cls, name, bases, attr)
    def wrapper(dumper, obj):
      return obj.__toyaml__(dumper)
    Dumper.add_representer(result, wrapper)
    return result

class Serializable(SerializableBase, metaclass=SerializableMeta):
  __slots__ = []

  def __getstate__(self):
    return collections.OrderedDict([(slot, getattr(self, slot, None)) for slot in self.__slots__])

  def __toyaml__(self, dumper):
    state = self.__getstate__()
    if isinstance(state, dict):
      return dumper.represent_mapping(self.__class__.__name__, state.items())
    elif isinstance(state, list):
      return dumper.represent_sequence(self.__class__.__name__, state, flow_style=False)
    elif isinstance(state, tuple):
      return dumper.represent_sequence(self.__class__.__name__, state, flow_style=True)
    else:
      return dumper.represent_scalar(self.__class__.__name__, state)

  def __repr__(self):
    #return "%s(%s)" % (self.__class__.__name__, self.to_dict())
    return yaml.dump(self, Dumper=Dumper)
    #return "%s(%s)" % (self.__class__.__name__, self.__getstate__())

  def __str__(self):
    return self.__repr__()




class CoordFrame(Serializable):
  __slots__ = ['labels']

  def __getstate__(self):
    return tuple(self.labels)

  def __init__(self, labels):
    self.labels = labels
  def __eq__(self, other):
    return id(self) == id(other)
  def __len__(self):
    return len(self.labels)
  def __getitem__(self, key):
    return self.labels[key]
  def __contains__(self, item):
    return self.labels.__contains__(item)
  def __iter__(self):
    return self.labels.__iter__()
  def __repr__(self):
    return "CoordFrame(%s)" % (repr(self.labels),)
  def subframe(self, s):
    return CoordFrame(self.labels[s])
  def slice(self, s):
    if isinstance(s, slice):
      return slice(self.labels.index(s.start) if s.start in self.labels else s.start,
                   self.labels.index(s.stop) if s.stop in self.labels else s.stop,
                   s.step)
    elif s in self.labels:
      return slice(self.labels.index(s), self.labels.index(s)+1, None)
    elif isinstance(s, numbers.Number):
      return slice(s, s+1, None)
    else:
      raise IndexError("Index out of range: %s" % (repr(s),))

  def key(self, k):
    if isinstance(k, slice):
      return self.slice(k)
    elif k in self.labels:
      return self.labels.index(k)
    elif isinstance(k, numbers.Number):
      return k
    else:
      raise IndexError("Index out of range: %s" % (repr(k),))



class MatrixFrame(Serializable):
  __slots__ = ['row_frame', 'col_frame']

  def __init__(self, row_frame, col_frame):
    self.row_frame = row_frame
    self.col_frame = col_frame
  def __getitem__(self, index):
    return (self.row_frame, self.col_frame)[index]
  def __eq__(self, other):
    return self.row_frame == other.row_frame and self.col_frame == other.col_frame


class CoordMatrix(Serializable):
  __slots__ = ['frame', 'matrix']
  def __getstate__(self):
    return collections.OrderedDict([('frame', self.frame), ('matrix', repr(self.matrix))])

  @property
  def row_frame(self):
    return self.frame[0]
  @property
  def col_frame(self):
    return self.frame[1]
  @property
  def rows(self):
    return len(self.row_frame) if self.row_frame is not None else 1
  @property
  def cols(self):
    return len(self.col_frame) if self.col_frame is not None else 1
  @property
  def shape(self):
    return (self.rows, self.cols)



  @staticmethod
  def eye(frame):
    return CoordMatrix(frame, frame, sympy.eye(len(frame)).evalf())
  @staticmethod
  def zeros(row_frame, col_frame):
    nrows = len(row_frame) if row_frame is not None else 1
    ncols = len(col_frame) if col_frame is not None else 1
    return CoordMatrix(row_frame, col_frame, sympy.zeros(nrows, ncols).evalf())
  @staticmethod
  def diag(frame, entries):
    return CoordMatrix(frame, frame, sympy.Matrix.diag(entries).evalf())


  def __init__(self, row_frame, col_frame, components):
    #trace("CoordMatrix.__init__: self=%x, row_frame=%s, col_frame=%s, components=%s" % (id(self), repr(row_frame), repr(col_frame), repr(list(components))))
    self.frame = MatrixFrame(row_frame, col_frame)
    #self.row_frame = row_frame
    #self.col_frame = col_frame
    self.matrix = sympy.Matrix(self.rows, self.cols, components)

  def __setitem__(self, key, value):
    if isinstance(key, tuple):
      row_key, col_key = key
      if isinstance(row_key, slice) or isinstance(col_key, slice):
        row_slice = self.row_frame.slice(row_key)
        col_slice = self.col_frame.slice(col_key)
        self.matrix.__setitem__((row_slice, col_slice), value)
      else:
        self.matrix.__setitem__((self.row_frame.key(row_key), self.col_frame.key(col_key)), value)
    elif key in self.row_frame:
      self.matrix.__setitem__(self.row_frame.key(key), value)
    elif isinstance(key, numbers.Number):
      self.matrix.__setitem__(key, value)
    else:
      raise IndexError("Index out of range: %s" % (repr(key),))
  def __getitem__(self, key):
    #trace("CoordMatrix.__getitem__: key=%s" % (repr(key),))
    if isinstance(key, tuple):
      row_key, col_key = key
      if isinstance(row_key, slice) or isinstance(col_key, slice):
        row_slice = self.row_frame.slice(row_key)
        col_slice = self.col_frame.slice(col_key)
        return CoordMatrix(self.row_frame.subframe(row_slice), self.col_frame.subframe(col_slice), self.matrix.__getitem__((row_slice, col_slice)))
      else:
        return self.matrix.__getitem__((self.row_frame.key(row_key), self.col_frame.key(col_key)))
    elif key in self.row_frame:
      return self.matrix.__getitem__(self.row_frame.key(key))
    elif isinstance(key, numbers.Number):
      return self.matrix.__getitem__(key)
    else:
      raise IndexError("Index out of range: %s" % (repr(key),))
  def __len__(self):
    return self.matrix.__len__()

  def __add__(self, other):
    if self.frame != other.frame:
      raise sympy.matrices.common.ShapeError("Matrix size mismatch: %s + %s" % (repr(self.frame), repr(other.frame)))
    return CoordMatrix(self.row_frame, self.col_frame, self.matrix.__add__(other.matrix))
  def __sub__(self, other):
    if self.frame != other.frame:
      raise sympy.matrices.common.ShapeError("Matrix size mismatch: %s - %s" % (repr(self.frame), repr(other.frame)))
    return CoordMatrix(self.row_frame, self.col_frame, self.matrix.__sub__(other.matrix))
  def __mul__(self, other):
    if isinstance(other, numbers.Number):
      return CoordMatrix(self.row_frame, self.col_frame, self.matrix.__mul__(other))
    elif isinstance(other, CoordMatrix) and self.col_frame != other.row_frame:
      raise sympy.matrices.common.ShapeError("Matrix size mismatch: (%d, %d) * (%d, %d)" % (self.rows, self.cols, other.rows, other.cols))
    return CoordMatrix(self.row_frame, other.col_frame, self.matrix.__mul__(other.matrix))

  def is_colvec(self):
    return self.row_frame is not None and self.col_frame is None
  def is_rowvec(self):
    return self.row_frame is None and self.col_frame is not None

  def _subs(self, sublist):
    allsubs = list()
    for lhs, rhs in sublist:
      if isinstance(lhs, CoordMatrixSymbol):
        if not (lhs.row_frame == rhs.row_frame and lhs.col_frame == rhs.col_frame):
          raise sympy.matrices.common.ShapeError("Substitution mismatch: lhs=%s, rhs=%s" % (repr(lhs.frame), repr(rhs.frame)))
        allsubs.extend([(lhs[idx], rhs[idx]) for idx in range(len(lhs))])
      elif isinstance(lhs, sympy.Symbol):
        allsubs.append((lhs, rhs))
    return CoordMatrix(self.row_frame, self.col_frame, self.matrix.subs(allsubs))
  def subs(self, *args, **kwargs):
    if len(args) == 2:
      return self._subs([(args[0], args[1])])
    elif len(args) == 1 and isinstance(args[0], dict):
      return self._subs(args[0].items())
    elif len(args) == 1 and isinstance(args[0], list):
      return self._subs(args[0])

  def evalf(self, *args, **kwargs):
    return CoordMatrix(self.row_frame, self.col_frame, self.matrix.evalf(*args, **kwargs))
  def transpose(self):
    return CoordMatrix(self.col_frame, self.row_frame, self.matrix.transpose())
  def inv(self):
    return CoordMatrix(self.col_frame, self.row_frame, self.matrix.inv())

  def jacobian(self, x):
    if self.is_colvec() and x.is_colvec():
      return CoordMatrix(self.row_frame, x.row_frame, self.matrix.jacobian(x.matrix))
    elif self.is_colvec() and x.is_rowvec():
      return CoordMatrix(self.row_frame, x.col_frame, self.matrix.jacobian(x.matrix))
    elif self.is_rowvec() and x.is_colvec():
      return CoordMatrix(self.col_frame, x.row_frame, self.matrix.jacobian(x.matrix))
    elif self.is_rowvec() and x.is_rowvec():
      return CoordMatrix(self.col_frame, x.col_frame, self.matrix.jacobian(x.matrix))
    else:
      raise sympy.matrices.common.ShapeError("Matrix size mismatch: (%d, %d) * (%d, %d)" % (self.rows, self.cols, x.rows, x.cols))



class CoordMatrixSymbol(CoordMatrix):
  __slots__ = ['name']

  def __getstate__(self):
    result = collections.OrderedDict()
    result.update({'name': self.name})
    result.update(super().__getstate__())
    return result

  def __init__(self, name, row_frame, col_frame):
    #trace("CoordMatrixSymbol.__init__: self=%x, name=%s, row_frame=%s, col_frame=%s" % (id(self), repr(name), repr(row_frame), repr(col_frame)))
    if row_frame is not None and col_frame is not None:
      super().__init__(row_frame, col_frame, [sympy.Symbol("%s.%s.%s" % (name, row_lbl, col_lbl)) for row_lbl in row_frame for col_lbl in col_frame])
    elif row_frame is not None and col_frame is None:
      super().__init__(row_frame, col_frame, [sympy.Symbol("%s.%s" % (name, row_lbl)) for row_lbl in row_frame])
    elif row_frame is None and col_frame is not None:
      super().__init__(row_frame, col_frame, [sympy.Symbol("%s.%s" % (name, col_lbl)) for col_lbl in col_frame])
    else:
      super().__init__(row_frame, col_frame, [sympy.Symbol("%s" % (name,))])
    self.name = name


class CoordVector(CoordMatrix):
  @staticmethod
  def zeros(frame):
    return CoordVector(frame, sympy.zeros(len(frame), len(g_one_dim)).evalf())

  def __init__(self, frame, components):
    trace("CoordVector.__init__: self=%x, frame=%s, components=%s" % (id(self), repr(frame), repr(list(components))))
    #super().__init__(frame, g_one_dim, components)
    super().__init__(frame, None, components)
    self.frame = frame
  #def jacobian(self, x):
  #  return CoordMatrix(self.frame, x.frame, self.matrix.jacobian(x.matrix))


class CoordVectorSymbol(CoordVector):
  def __init__(self, name, frame):
    trace("CoordVectorSymbol.__init__: self=%x, name=%s, frame=%s" % (id(self), repr(name), repr(frame)))
    super().__init__(frame, [sympy.Symbol("%s.%s" % (name, lbl)) for lbl in frame])
    self.name = name
  def subs(self, vec):
    return {self.__getitem__(idx): vec[idx] for idx in range(len(self))}





class EKFDesc(Serializable):
  __slots__ = ['x', 'u', 'z', 'f', 'h', 'c', 'dfdx', 'dhdx']

  def __init__(self, x, u, z, f, h, c):
    self.x = x
    self.u = u
    self.z = z
    self.f = f
    self.h = h
    self.c = c
    self.dfdx = self.f.jacobian(self.x)
    self.dhdx = self.h.jacobian(self.x)

  def make_diag_Q(self, diag):
    return CoordMatrix.diag(self.x.row_frame, diag)
  def make_diag_R(self, diag):
    return CoordMatrix.diag(self.z.row_frame, diag)

class EKF(Serializable):
  __slots__ = ['desc', 'k', 'x', 'P', 'c', 'Q_k', 'R_k', 'u_k', 'f_k', 'F_k', 'x_pred', 'P_pred', 'z_k', 'h_k', 'H_k', 'y_k', 'S_k', 'K_k', 'x_est', 'P_est',
               'subs_pred', 'subs_update']

  def __init__(self, desc, x=None, P=None, c=None, Q=None, R=None):
    self.desc = desc

    self.k = 0
    self.x = CoordMatrix.zeros(self.desc.x.row_frame, None)
    self.P = CoordMatrix.zeros(self.desc.x.row_frame, self.desc.x.row_frame)
    self.c = CoordMatrix.zeros(self.desc.c.row_frame, None)
    self.Q_k = CoordMatrix.zeros(self.desc.x.row_frame, self.desc.x.row_frame)
    self.R_k = CoordMatrix.zeros(self.desc.z.row_frame, self.desc.z.row_frame)


  def predict(self, u_k, Q_k=None):
    self.k += 1

    self.u_k = CoordMatrix(self.desc.u.row_frame, None, u_k)
    if Q_k is not None:
      self.Q_k = CoordMatrix(self.desc.x.row_frame, self.desc.x.row_frame, Q_k)

    self.subs_pred = [(self.desc.c, self.c), (self.desc.x, self.x), (self.desc.u, self.u_k)]

    self.f_k = self.desc.f.subs(self.subs_pred).evalf()
    self.F_k = self.desc.dfdx.subs(self.subs_pred).evalf()

    self.x_pred = self.f_k
    self.P_pred = (self.F_k * self.P * self.F_k.transpose() + self.Q_k)


  def update(self, z_k, R_k=None):
    self.z_k = CoordMatrix(self.desc.z.row_frame, None, z_k)
    if R_k is not None:
      self.R_k = CoordMatrix(self.desc.z.row_frame, self.desc.z.row_frame, R_k)

    self.subs_update = [(self.desc.c, self.c), (self.desc.x, self.x_pred)]

    self.h_k = self.desc.h.subs(self.subs_update).evalf()
    self.H_k = self.desc.dhdx.subs(self.subs_update).evalf()

    self.y_k = self.z_k - self.h_k
    self.S_k = self.H_k * self.P_pred * self.H_k.transpose() + self.R_k
    self.K_k = self.P_pred * self.H_k.transpose() * self.S_k.inv()

    self.x_est = self.x_pred + self.K_k * self.y_k
    self.P_est = (CoordMatrix.eye(self.desc.x.row_frame) - self.K_k * self.H_k) * self.P_pred

    self.x = self.x_est.evalf()
    self.P = self.P_est.evalf()









class Vehicle(Serializable):
  __slots__ = ['props', 'state', 'ekf']

  class Properties(Serializable):
    __slots__ = ['wheel_base', 'ekf_dt', 'brake_coeff']

    def __init__(self, wheel_base=1.0, ekf_dt=0.1, brake_coeff=0.8):
      self.wheel_base = float(wheel_base)
      self.ekf_dt = float(ekf_dt)
      self.brake_coeff = float(brake_coeff)

  class State(Serializable):
    __slots__ = ['t', 'x', 'y', 'hdg', 'vel', 'accel', 'throttle', 'brake', 'steering_angle', 'steering_rate', 'yaw_rate']

    def __init__(self, t=0.0, x=0.0, y=0.0, hdg=0.0, vel=0.0, accel=0.0, throttle=0.0, brake=0.0, steering_angle=0.0, steering_rate=0.0, yaw_rate=0.0):
      self.t = float(t)
      self.x = float(x)
      self.y = float(y)
      self.hdg = float(hdg)
      self.vel = float(vel)
      self.accel = float(accel)
      self.throttle = float(throttle)
      self.brake = float(brake)
      self.steering_angle = float(steering_angle)
      self.steering_rate = float(steering_rate)
      self.yaw_rate = float(yaw_rate)


  def update_ekf(self):
    u_k = [self.state.throttle, self.state.brake, self.state.steering_rate]
    Q_k = self.ekf.desc.make_diag_Q([0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])

    self.ekf.predict(u_k, Q_k)

    z_k = [
      math.sqrt(self.state.x*self.state.x + self.state.y*self.state.y),
      math.atan2(-self.state.y, -self.state.x) - self.state.hdg,
      self.state.vel,
      self.state.vel * math.tan(self.state.steering_angle) / self.props.wheel_base,
    ]
    R_k = self.ekf.desc.make_diag_R([0.01, 0.01, 0.01, 0.01])

    self.ekf.update(z_k, R_k)


    print("Vehicle.update_ekf: self.ekf.z_k=", repr(self.ekf.z_k.matrix.transpose()))
    print("Vehicle.update_ekf: self.ekf.h_k=", repr(self.ekf.h_k.matrix.transpose()))
    print("Vehicle.update_ekf: self.ekf.x=", repr(self.ekf.x.matrix.transpose()))
    print("Vehicle.update_ekf: self.ekf.P=", repr(self.ekf.P.matrix))

  def step(self, delta_t):
    curr_state = self.state

    next_state = Vehicle.State()
    next_state.yaw_rate = curr_state.vel * math.tan(curr_state.steering_angle) / self.props.wheel_base
    next_state.steering_rate = curr_state.steering_rate
    next_state.steering_angle = curr_state.steering_angle + delta_t * curr_state.steering_rate
    next_state.throttle = curr_state.throttle
    next_state.brake = curr_state.brake
    next_state.accel = curr_state.throttle - (1.0 - self.props.brake_coeff*delta_t) * curr_state.brake * curr_state.vel
    next_state.vel = curr_state.vel + delta_t * curr_state.accel
    next_state.hdg = curr_state.hdg + delta_t * curr_state.yaw_rate
    next_state.x = curr_state.x + delta_t * curr_state.vel * math.cos(curr_state.hdg)
    next_state.y = curr_state.y + delta_t * curr_state.vel * math.sin(curr_state.hdg)
    next_state.t = curr_state.t + delta_t

    self.state = next_state

    print("Vehicle.step: self.state=", self.state)

    inv_radius = math.tan(self.state.steering_angle) / self.props.wheel_base
    print("vehicle.step: inv_radius=", inv_radius)
    print("vehicle.step: radius=", 1.0 / inv_radius if math.fabs(inv_radius) > 0.001 else 0.0)


    while self.state.t > self.ekf.x['t'] + self.ekf.c['dt']:
      self.update_ekf()


  def __init__(self, props=Properties(wheel_base=1.0), state=State(t=0.0, x=1.0)):
    self.props = props
    self.state = state

    ekf_x_coords = CoordFrame(['t', 'x', 'y', 'theta', 'vel', 'accel', 'delta'])
    ekf_u_coords = CoordFrame(['throttle', 'brake', 'steering_rate'])
    ekf_z_coords = CoordFrame(['dist', 'hdg', 'speed', 'gyro'])
    ekf_c_coords = CoordFrame(['dt', 'L', 'brake_coeff'])

    ekf_x = CoordMatrixSymbol('x', ekf_x_coords, None)
    ekf_u = CoordMatrixSymbol('u', ekf_u_coords, None)
    ekf_z = CoordMatrixSymbol('z', ekf_z_coords, None)
    ekf_c = CoordMatrixSymbol('c', ekf_c_coords, None)
    ekf_f = CoordMatrix(ekf_x_coords, None, [
      ekf_x['t'] + ekf_c['dt'],
      ekf_x['x'] + sympy.cos(ekf_x['theta'])*ekf_x['vel']*ekf_c['dt'] + 0.5*sympy.cos(ekf_x['theta'])*ekf_x['accel']*ekf_c['dt']*ekf_c['dt'],
      ekf_x['y'] + sympy.sin(ekf_x['theta'])*ekf_x['vel']*ekf_c['dt'] + 0.5*sympy.sin(ekf_x['theta'])*ekf_x['accel']*ekf_c['dt']*ekf_c['dt'],
      ekf_x['theta'] + ekf_c['dt']*ekf_x['vel']*sympy.tan(ekf_x['delta'])/ekf_c['L'],
      ekf_x['vel'] + ekf_c['dt']*ekf_x['accel'],
      ekf_u['throttle'] - (1.0 - ekf_c['brake_coeff']*ekf_c['dt'])*ekf_u['brake']*ekf_x['vel'],
      ekf_x['delta'] + ekf_c['dt']*ekf_u['steering_rate']
    ])
    ekf_h = CoordMatrix(ekf_z_coords, None, [
      sympy.sqrt(ekf_x['x']*ekf_x['x'] + ekf_x['y']*ekf_x['y']),
      sympy.atan2(-ekf_x['y'], -ekf_x['x']) - ekf_x['theta'],
      ekf_x['vel'],
      ekf_x['vel']*sympy.tan(ekf_x['delta'])/ekf_c['L']
    ])

    ekf_desc = EKFDesc(ekf_x, ekf_u, ekf_z, ekf_f, ekf_h, ekf_c)
    self.ekf = EKF(ekf_desc)



    self.ekf.x['t'] = self.state.t
    self.ekf.x['x'] = self.state.x
    self.ekf.x['y'] = self.state.y
    self.ekf.x['theta'] = self.state.hdg
    self.ekf.x['vel'] = self.state.vel
    self.ekf.x['accel'] = self.state.accel
    self.ekf.x['delta'] = self.state.steering_angle


    self.ekf.c['dt'] = self.props.ekf_dt
    self.ekf.c['L'] = self.props.wheel_base
    self.ekf.c['brake_coeff'] = self.props.brake_coeff



class Rate:
  def __init__(self, hz):
    self.period = 1.0 / float(hz)
    self.start = time.time()
    self.count = 0
  def remaining(self):
    curr_time = time.time()
    next_time = self.start + self.period * (self.count + 1)
    return next_time - curr_time
  def sleep(self):
    time_to_sleep = self.remaining()
    while time_to_sleep > 0.0:
      time.sleep(time_to_sleep)
      time_to_sleep = self.remaining()
    self.count += 1
    return True



def test_vehicle():
  delta_t = 0.01

  vehicle = Vehicle(props=Vehicle.Properties(wheel_base=1.0, ekf_dt=0.05), state=Vehicle.State(t=0.0, x=1.0, y=0.1, hdg=0.0, vel=1.0))

  print("test_vehicle: vehicle.ekf.x=", repr(vehicle.ekf.x.matrix.transpose()))
  print("test_vehicle: vehicle.ekf.P=", repr(vehicle.ekf.P.matrix))
  print("test_vehicle: vehicle.state=", vehicle.state)


  rate = Rate(0.5)
  while True:
    line = sys.stdin.readline().strip()
    if line == 'r':
      vehicle.state.steering_rate -= 0.1
    elif line == 'l':
      vehicle.state.steering_rate += 0.1
    elif line == 's':
      vehicle.state.steering_rate = 0.0
      vehicle.state.steering_angle = 0.0
    elif line == 'T':
      vehicle.state.throttle = 1.0
    elif line == 't':
      vehicle.state.throttle = 0.0
    elif line == 'B':
      vehicle.state.brake = 1.0
    elif line == 'b':
      vehicle.state.brake = 0.0
    else:
      pass

    vehicle.step(delta_t)
    #rate.sleep()


def main():
  test_vehicle()

if __name__ == '__main__':
  main()
