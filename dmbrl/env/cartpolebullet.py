from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np

import pybullet
from pybullet_envs.scene_abstract import SingleRobotEmptyScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.robot_pendula import InvertedPendulum

class CartpoleBulletEnv(MJCFBaseBulletEnv):

  def __init__(self):
    self.robot = InvertedPendulum()
    MJCFBaseBulletEnv.__init__(self, self.robot)
    self.stateId = -1

  def create_single_player_scene(self, bullet_client):
    return SingleRobotEmptyScene(bullet_client, gravity=9.8, timestep=0.0165, frame_skip=1)

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
      reward = np.cos(self.robot.theta)
      done = False
    else:
      reward = 1.0
      done = np.abs(self.robot.theta) > .2
    self.rewards = [float(reward)]
    self.HUD(state, a, done)
    return state, sum(self.rewards), done, {}

  def camera_adjust(self):
    self.camera.move_and_look_at(0, 1.2, 1.0, 0, 0, 0.5)