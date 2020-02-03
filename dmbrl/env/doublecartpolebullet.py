from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np

import pybullet
from pybullet_envs.scene_abstract import SingleRobotEmptyScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.robot_bases import MJCFBasedRobot

class DoubleCartpoleBulletEnv(MJCFBaseBulletEnv):

  def __init__(self):
    self.robot = InvertedDoublePendulumModified()
    MJCFBaseBulletEnv.__init__(self, self.robot)
    self.stateId = -1

  def create_single_player_scene(self, bullet_client):
    return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.05, frame_skip=1)

  def reset(self):
    if (self.stateId >= 0):
      self._p.restoreState(self.stateId)
    r = MJCFBaseBulletEnv.reset(self)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()
    return r

  def step(self, a):
    self.robot.apply_action(a)
    self.scene.global_step()
    state = self.robot.calc_state()  # sets self.pos_x self.pos_y
    # upright position: 0.6 (one pole) + 0.6 (second pole) * 0.5 (middle of second pole) = 0.9
    # using <site> tag in original xml, upright position is 0.6 + 0.6 = 1.2, difference +0.3
    dist_penalty = 0.01 * self.robot.pos_x**2 + (self.robot.pos_y + 0.3 - 2)**2
    # v1, v2 = self.model.data.qvel[1:3]   TODO when this fixed https://github.com/bulletphysics/bullet3/issues/1040
    #vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
    vel_penalty = 0
    alive_bonus = 10
    done = self.robot.pos_y + 0.3 <= 1
    self.rewards = [float(alive_bonus), float(-dist_penalty), float(-vel_penalty)]
    self.HUD(state, a, done)
    return state, sum(self.rewards), done, {}

  def camera_adjust(self):
    self.camera.move_and_look_at(0, 1.2, 1.0, 0, 0, 0.5)

  def render(self, mode='rgb_array', close=False):
    if mode != "rgb_array":
      return np.array([])
    return super().render(mode,close).astype('uint8')

class InvertedDoublePendulumModified(MJCFBasedRobot):

  def __init__(self):
    MJCFBasedRobot.__init__(self, 'inverted_double_pendulum.xml', 'cart', action_dim=1, obs_dim=6)

  def robot_specific_reset(self, bullet_client):
    self._p = bullet_client
    self.pole2 = self.parts["pole2"]
    self.slider = self.jdict["slider"]
    self.j1 = self.jdict["hinge"]
    self.j2 = self.jdict["hinge2"]
    u = self.np_random.uniform(low=-.1, high=.1, size=[2])
    self.j1.reset_current_position(float(u[0]), 0)
    self.j2.reset_current_position(float(u[1]), 0)
    self.j1.set_motor_torque(0)
    self.j2.set_motor_torque(0)

  def apply_action(self, a):
    assert (np.isfinite(a).all())
    if not np.isfinite(a).all():
      print("a is inf")
      a[0] = 0
    self.slider.set_motor_torque(100 * float(np.clip(a[0], -1, +1)))

  def calc_state(self):
    theta, theta_dot = self.j1.current_position()
    gamma, gamma_dot = self.j2.current_position()
    x, vx = self.slider.current_position()
    self.pos_x, _, self.pos_y = self.pole2.pose().xyz()
    assert (np.isfinite(x))
    
    if not np.isfinite(x):
      print("x is inf")
      x = 0
      
    if not np.isfinite(vx):
      print("vx is inf")
      vx = 0
      
    if not np.isfinite(theta):
      print("theta is inf")
      theta = 0
      
    if not np.isfinite(theta_dot):
      print("theta_dot is inf")
      theta_dot = 0

    if not np.isfinite(gamma):
      print("gamma is inf")
      gamma = 0

    if not np.isfinite(gamma_dot):
      print("gamma_dot is inf")
      gamma_dot = 0

    return np.array([
        x,
        theta,
        gamma,
        vx,
        theta_dot,
        gamma_dot,
    ])