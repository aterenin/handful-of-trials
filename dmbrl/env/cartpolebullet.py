from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np

import pybullet
from pybullet_envs.scene_abstract import SingleRobotEmptyScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.robot_bases import MJCFBasedRobot

class CartpoleBulletEnv(MJCFBaseBulletEnv):

  def __init__(self):
    self.robot = InvertedPendulumModified()
    MJCFBaseBulletEnv.__init__(self, self.robot)
    self.stateId = -1

  def create_single_player_scene(self, bullet_client):
    return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.05, frame_skip=1)

  def reset(self):
    if (self.stateId >= 0):
      #print("InvertedPendulumBulletEnv reset p.restoreState(",self.stateId,")")
      self._p.restoreState(self.stateId)
    r = MJCFBaseBulletEnv.reset(self)
    if (self.stateId < 0):
      self.stateId = self._p.saveState()
      #print("InvertedPendulumBulletEnv reset self.stateId=",self.stateId)
    return r

  def step(self, a):
    self.robot.apply_action(a)
    self.scene.global_step()
    state = self.robot.calc_state()  # sets self.pos_x self.pos_y
    vel_penalty = 0
    if self.robot.swingup:
      reward = np.cos(state[1])
      done = np.abs(state[0]) > 4.99
    else:
      reward = 1.0
      done = np.abs(state[1]) > .2
    self.rewards = [float(reward)]
    self.HUD(state, a, done)
    return state, sum(self.rewards), done, {}

  def camera_adjust(self):
    self.camera.move_and_look_at(0, 1.2, 1.0, 0, 0, 0.5)

  def render(self, mode='rgb_array', close=False):
    if mode != "rgb_array":
      return np.array([])
    return super().render(mode,close).astype('uint8')

class InvertedPendulumModified(MJCFBasedRobot):
	swingup = True
	def __init__(self):
		dir_path = os.path.dirname(os.path.realpath(__file__))
		MJCFBasedRobot.__init__(self, '%s/assets/inverted_pendulum.xml' % dir_path, 'cart', action_dim=1, obs_dim=4)

	def robot_specific_reset(self, bullet_client):
		self._p = bullet_client
		self.pole = self.parts["pole"]
		self.slider = self.jdict["slider"]
		self.j1 = self.jdict["hinge"]
		u = self.np_random.uniform(low=-.1, high=.1)
		self.j1.reset_current_position( u if not self.swingup else 3.1415+u , 0)
		self.j1.set_motor_torque(0)

	def apply_action(self, a):
		assert( np.isfinite(a).all() )
		if not np.isfinite(a).all():
			print("a is inf")
			a[0] = 0
		self.slider.set_motor_torque(  100*float(np.clip(a[0], -1, +1)) )

	def calc_state(self):
		theta, theta_dot = self.j1.current_position()
		x, vx = self.slider.current_position()
		assert( np.isfinite(x) )

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

		return np.array([
			x, 
      theta, 
      vx, 
      theta_dot
			])