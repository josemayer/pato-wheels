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
from connections import get_motor_inner_left_matrix, get_motor_inner_right_matrix, get_motor_outer_left_matrix, get_motor_outer_right_matrix
from preprocessing import preprocess
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

        self.l_max = -math.inf
        self.r_max = -math.inf
        self.l_min = math.inf
        self.r_min = math.inf
        self.left_inner = None
        self.right_inner = None
        self.left_outer = None
        self.right_outer = None

        self.img = self.env.front()
        self.img_shape = self.img.shape[0], self.img.shape[1]

        self.score = 0

        key_handler = key.KeyStateHandler()
        env.unwrapped.window.push_handlers(key_handler)
        self.key_handler = key_handler

    def get_pwm_control(self, v: float, w: float)-> (float, float):
        ''' Takes velocity v and angle w and returns left and right power to motors.'''
        V_l = (self.motor_gain - self.motor_trim)*(v-w*self.baseline)/self.radius
        V_r = (self.motor_gain + self.motor_trim)*(v+w*self.baseline)/self.radius
        return V_l, V_r

    def send_commands(self, dt: float):
        velocity = 0
        rotation = 0
        
        img = self.env.front() 
        
        if img is None:
            return 0.0, 0.0

        if self.left_inner is None:
            # if it is the first time, we initialize the structures
            shape = self.img_shape
            self.left_inner = get_motor_inner_left_matrix(shape)
            self.right_inner = get_motor_inner_right_matrix(shape)
            self.left_outer = get_motor_outer_left_matrix(shape)
            self.right_outer = get_motor_outer_right_matrix(shape)


        # let's take only the intensity of IMG
        P, Q = preprocess(img)
        # now we just compute the activation of our sensors
        l = float(np.sum(P * self.left_inner)) + float(np.sum(Q * self.left_outer))
        r = float(np.sum(P * self.right_inner)) + float(np.sum(Q * self.right_outer))

        # These are big numbers -- we want to normalize them.
        # We normalize them using the history

        # first, we remember the high/low of these raw signals
        self.l_max = max(l, self.l_max)
        self.r_max = max(r, self.r_max)
        self.l_min = min(l, self.l_min)
        self.r_min = min(r, self.r_min)

        # now rescale from 0 to 1
        ls = rescale(l, self.l_min, self.l_max)
        rs = rescale(r, self.r_min, self.r_max)

        gain = self.motor_gain
        const = 0.2
        pwm_left = const + ls * gain
        pwm_right = const + rs * gain
        
        # run image processing routines
        #P, Q, M = self.preprocess(img) # returns inner, outter and combined mask matrices
        print('>', pwm_left, pwm_right) # uncomment for debugging
        # Now send command to motors

        #if self.key_handler[key.W]:
        #    velocity += 0.5
        #if self.key_handler[key.A]:
        #    rotation += 1.5
        #if self.key_handler[key.S]:
        #    velocity -= 0.5
        #if self.key_handler[key.D]:
        #    rotation -= 1.5

        #pwm_left, pwm_right = self.get_pwm_control(velocity, rotation)
        _, r, _, _ = self.env.step(pwm_left, pwm_right)
        
        self.score += (r-self.score)/self.env.step_count
        self.env.render("human", text = f", score: {self.score:.3f}") 

def rescale(a: float, L: float, U: float):
    if np.allclose(L, U):
        return 0.0
    return (a - L) / (U - L)
