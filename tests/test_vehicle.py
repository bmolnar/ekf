import sys
import time
import math
from ekf import *
import ekf.math

class Serializable(object):
  def to_dict(self):
    pass


class VehicleHandler:
  def on_init(self, obj):
    pass
  def on_step(self, obj):
    pass
  def on_ekf_update(self, obj):
    pass

class VehicleCallbacks(VehicleHandler):
  def __init__(self, on_init=None, on_step=None, on_ekf_update=None):
    self._on_init = on_init
    self._on_step = on_step
    self._on_ekf_update = on_ekf_update
  def on_init(self, obj):
    if self._on_init is not None:
      self._on_init(obj)
  def on_step(self, obj):
    if self._on_step is not None:
      self._on_step(obj)
  def on_ekf_update(self, obj):
    if self._on_ekf_update is not None:
      self._on_ekf_update(obj)



class Vehicle(Serializable):
  __slots__ = ['props', 'state', 'ekf', 'handler']

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

  def ekf_init(self):
    xf = CoordFrame(('t', 'x', 'y', 'theta', 'vel', 'accel', 'delta'))
    uf = CoordFrame(('throttle', 'brake', 'steering_rate'))
    zf = CoordFrame(('dist', 'hdg', 'speed', 'gyro'))
    cf = CoordFrame(('dt', 'L', 'brake_coeff'))

    x = CoordVectorSymbol('x', xf)
    u = CoordVectorSymbol('u', uf)
    z = CoordVectorSymbol('z', zf)
    c = CoordVectorSymbol('c', cf)
    f = CoordVector(xf,
    [
      x['t'] + c['dt'],
      x['x'] + ekf.math.cos(x['theta'])*x['vel']*c['dt'] + 0.5*ekf.math.cos(x['theta'])*x['accel']*c['dt']*c['dt'],
      x['y'] + ekf.math.sin(x['theta'])*x['vel']*c['dt'] + 0.5*ekf.math.sin(x['theta'])*x['accel']*c['dt']*c['dt'],
      x['theta'] + c['dt']*x['vel']*ekf.math.tan(x['delta'])/c['L'],
      x['vel'] + c['dt']*x['accel'],
      u['throttle'] - (1.0 - c['brake_coeff']*c['dt'])*u['brake']*x['vel'],
      x['delta'] + c['dt']*u['steering_rate']
    ])
    h = CoordVector(zf,
    [
      ekf.math.sqrt(x['x']*x['x'] + x['y']*x['y']),
      ekf.math.atan2(-x['y'], -x['x']) - x['theta'],
      x['vel'],
      x['vel']*ekf.math.tan(x['delta'])/c['L']
    ])

    ekf_desc = EKFDesc(x, u, z, f, h, c)

    self.ekf = ekf.EKF(ekf_desc)        
    self.ekf.x_k = self.ekf.desc.make_x([self.state.t, self.state.x, self.state.y, self.state.hdg, self.state.vel, self.state.accel, self.state.steering_angle])
    self.ekf.c_k = self.ekf.desc.make_c([self.props.ekf_dt, self.props.wheel_base, self.props.brake_coeff])

  def ekf_update(self):
    control = [self.state.throttle, self.state.brake, self.state.steering_rate]

    measurement = [
        math.sqrt(self.state.x*self.state.x + self.state.y*self.state.y),
        math.atan2(-self.state.y,-self.state.x) - self.state.hdg,
        self.state.vel,
        self.state.vel * math.tan(self.state.steering_angle) / self.props.wheel_base,
    ]

    u_k = self.ekf.desc.make_u(control)
    Q_k = self.ekf.desc.make_diag_Q([0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
    self.ekf.predict(u_k, Q_k)

    z_k = self.ekf.desc.make_z(measurement)
    R_k = self.ekf.desc.make_diag_R([0.01, 0.01, 0.01, 0.01])
    self.ekf.update(z_k, R_k)

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
    self.handler.on_step(self)

    #print("Vehicle.step: self.state=", self.state)
    #inv_radius = math.tan(self.state.steering_angle) / self.props.wheel_base
    #print("vehicle.step: inv_radius=", inv_radius)
    #print("vehicle.step: radius=", 1.0 / inv_radius if math.fabs(inv_radius) > 0.001 else 0.0)

    while self.state.t > (self.ekf.x_k[self.ekf.desc.xf.index('t')] + self.ekf.c_k[self.ekf.desc.cf.index('dt')]):
      self.ekf_update()
      self.handler.on_ekf_update(self)

  def __init__(self, props=Properties(wheel_base=1.0), state=State(t=0.0, x=1.0), handler=VehicleCallbacks()):
    self.props = props
    self.state = state
    self.handler = handler
    self.ekf_init()
    self.handler.on_init(self)





def test_vehicle():
  delta_t = 0.01

  def on_init(vehicle):
    vs = vehicle.state
    vp = vehicle.props
    inv_radius = math.tan(vs.steering_angle) / vp.wheel_base
    print("on_init: vehicle.state: t=%f, x=%f, y=%f, hdg=%f, vel=%f, accel=%f, throttle=%f, brake=%f, steering_angle=%f, steering_rate=%f, yaw_rate=%f, inv_radius=%f" % (vs.t, vs.x, vs.y, vs.hdg, vs.vel, vs.accel, vs.throttle, vs.brake, vs.steering_angle, vs.steering_rate, vs.yaw_rate, inv_radius))
    print("on_init: vehicle.ekf.x_k:", repr(vehicle.ekf.x_k.transpose()))

  def on_step(vehicle):
    vs = vehicle.state
    vp = vehicle.props
    inv_radius = math.tan(vs.steering_angle) / vp.wheel_base
    print("on_step: vehicle.state: t=%f, x=%f, y=%f, hdg=%f, vel=%f, accel=%f, throttle=%f, brake=%f, steering_angle=%f, steering_rate=%f, yaw_rate=%f, inv_radius=%f" % (vs.t, vs.x, vs.y, vs.hdg, vs.vel, vs.accel, vs.throttle, vs.brake, vs.steering_angle, vs.steering_rate, vs.yaw_rate, inv_radius))

  def on_ekf_update(vehicle):
    print("on_ekf_update: vehicle.ekf.x_k:", repr(vehicle.ekf.x_k.transpose()))
    #print("on_ekf_update: vehicle.ekf.P_k:", repr(vehicle.ekf.P_k))

  callbacks = VehicleCallbacks()
  callbacks.on_init = on_init
  callbacks.on_step = on_step
  callbacks.on_ekf_update = on_ekf_update


  vehicle = Vehicle(props=Vehicle.Properties(wheel_base=1.0, ekf_dt=0.05), state=Vehicle.State(t=0.0, x=1.0, y=0.1, hdg=0.0, vel=1.0), handler=callbacks)

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


def main():
  test_vehicle()

if __name__ == '__main__':
  main()
