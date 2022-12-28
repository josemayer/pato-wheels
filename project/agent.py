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

        # load model of object detection
        self.model_od = tensorflow.keras.models.load_model('od.h5')
        
        # load model of dodge
        self.model_dodge = tensorflow.keras.models.load_model('ddg.h5')

        # load lane following model
        self.model_lf = tensorflow.keras.models.load_model('lf.h5')

        self.score = 0

        key_handler = key.KeyStateHandler()
        env.unwrapped.window.push_handlers(key_handler)
        self.key_handler = key_handler
        
        # Color segmentation hyperspace
        self.inner_lower = np.array([22, 93, 160])
        self.inner_upper = np.array([45, 255, 255])
        self.outer_lower = np.array([0, 0, 130])
        self.outer_upper = np.array([179, 85, 255])

    def preprocess(self, image):
        """ Returns a 2D array mask color segmentation of the image """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # obtain HSV representation of image
        # filter out dashed yellow "inner" line
        inner_mask = cv2.inRange(hsv, self.inner_lower, self.inner_upper)//255
        # filter out solid white "outer" line
        outer_mask = cv2.inRange(hsv, self.outer_lower, self.outer_upper)//255
        # Note: it is possible to filter out pixels in the RGB format
        #  by replacing `hsv` with `image` in the commands above
        # produces combined mask (might or might not be useful)
        mask = cv2.bitwise_or(inner_mask, outer_mask)
        self.masked = cv2.bitwise_and(image, image, mask=mask)
        return inner_mask, outer_mask

    def get_pwm_control(self, v: float, w: float)-> (float, float):
        ''' Takes velocity v and angle w and returns left and right power to motors.'''
        V_l = (self.motor_gain - self.motor_trim)*(v-w*self.baseline)/self.radius
        V_r = (self.motor_gain + self.motor_trim)*(v+w*self.baseline)/self.radius
        return V_l, V_r

    def send_commands(self, dt: float):
        ''' Agent control loop '''
        # acquire front camera image
        img = self.env.front()

        # run image processing routines
        P, Q = self.preprocess(img) # returns inner and outter mask matrices
 
        # transform image to shape (60, 80, 3)
        img_inference = cv2.resize(img, (80, 60))
       
        # predict single image img_inference
        img_inference = np.expand_dims(img_inference, axis=0)
        prediction = self.model_od.predict(img_inference, verbose=False)
        if prediction[0] > 0.5:
            # activate the dodge model
            prediction = self.model_dodge.predict(img_inference, verbose=False)
            pwm_left, pwm_right = self.get_pwm_control(prediction[0][0], prediction[0][1])
            self.env.step(pwm_left, pwm_right)
            return

        # resize masks P and Q to (60, 80)
        p_resized = cv2.resize(P, (80, 60))
        q_resized = cv2.resize(Q, (80, 60))

        # create a 2-channel image with the masks
        mask = np.zeros((60, 80, 2))
        mask[:, :, 0] = p_resized
        mask[:, :, 1] = q_resized
        
        # cut off the 30% top pixels
        mask = mask[(3 * mask.shape[0])//10:, :, :]

        masks = np.expand_dims(mask, axis=0)
        prediction = self.model_lf.predict(masks, verbose=False)
        pwm_left, pwm_right = self.get_pwm_control(prediction[0][0], prediction[0][1])
        self.env.step(pwm_left, pwm_right)

       #  for visualization
        self.env.render('human')
