# MAC0318 Intro to Robotics
# Please fill-in the fields below with every team member info
#
# Name:
# NUSP:
#
# Name:
# NUSP:
#
# Name:
# NUSP:
#
# Name:
# NUSP:
#
# Any supplemental material for your agent to work (e.g. neural networks, data, etc.) should be
# uploaded elsewhere and listed down below together with a download link.
#
#
#
# ---
#
# Final Project - The Travelling Mailduck Problem
#
# Don't forget to run this file from the Duckievillage root directory path (example):
#   cd ~/MAC0318/duckievillage
#   conda activate duckietown
#   python3 assignments/challenge/challenge.py assignments/challenge/examples/challenge_n
#
# Submission instructions:
#  0. Add your names and USP numbers to the file header above.
#  1. Make sure that any last change hasn't broken your code. If the code crashes without running you'll get a 0.
#  2. Submit this file via e-disciplinas.

import pyglet
from pyglet.window import key
import numpy as np
import math
import random
from duckievillage import create_env, FRONT_VIEW_MODE
import cv2
import tensorflow

class Agent:
    def __init__(self, env):
        self.env = env
        self.radius = 0.0318
        self.baseline = env.unwrapped.wheel_dist/2
        self.motor_gain = 0.68*0.0784739898632288
        self.motor_trim = 0.0007500911693361842
        self.initial_pos = env.get_position()
        
        self.score = 0

        key_handler = key.KeyStateHandler()
        env.unwrapped.window.push_handlers(key_handler)
        self.key_handler = key_handler

    def get_pwm_control(self, v: float, w: float)-> (float, float):
        ''' Takes velocity v and angle w and returns left and right power to motors.'''
        V_l = (self.motor_gain - self.motor_trim)*(v-w*self.baseline)/self.radius
        V_r = (self.motor_gain + self.motor_trim)*(v+w*self.baseline)/self.radius
        return V_l, V_r

    def preprocess(self):
        pass

    def send_commands(self, dt: float):
        velocity = 0
        rotation = 0

        if self.key_handler[key.W]:
            velocity += 0.5
        if self.key_handler[key.A]:
            rotation += 1.5
        if self.key_handler[key.S]:
            velocity -= 0.5
        if self.key_handler[key.D]:
            rotation -= 1.5

        pwm_left, pwm_right = self.get_pwm_control(velocity, rotation)
        _, r, _, _ = self.env.step(pwm_left, pwm_right)
        self.score += (r-self.score)/self.env.step_count
        self.env.render(text = f", score: {self.score:.3f}")
