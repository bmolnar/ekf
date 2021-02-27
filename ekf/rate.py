import time

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

