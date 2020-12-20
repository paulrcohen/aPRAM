#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 14:17:07 2020

@author: prcohen
"""
import numpy as np
import pandas as pd
from numpy.random import default_rng
rng = default_rng() # stuff for random sampling
import matplotlib.pyplot as plt


class Weather ():
    def __init__(self, num_locations, window_width = 5):
        # loc and scale are parameters of gumbel distribution
        # .12 and .05 result in .15"/day, about right for SRV in rainy season

        self.k = window_width
        self.num_locations = num_locations

        self.climate = [
            [0, 30, 365, 395, 5, .01],
            [30, 60, 395, 425, 8, .03],
            [60, 90, 425, 455, 12, .04],
            [90, 120, 455, 485, 20, .05],
            [120, 150, 485, 515, 25, .11],
            [150, 180, 515, 545, 27, .12],
            [180, 210, 545, 575, 32, .13],
            [210, 240, 575, 605, 30, .12],
            [240, 270, 605, 635, 25, .11],
            [270, 300, 635, 665, 20, .10],
            [300, 330, 665, 695, 15, .05],
            [330, 365, 695, 725, 8, .02]
            ]

        self.rain_window = np.array([self.rain_sample(0) for i in range(window_width)]).transpose()
        self.temp_window = np.array([self.temp_sample(0) for i in range(window_width)]).transpose()
        self.rain_moving_average() # sets self.avg_rain to an array of self.num_locations of rainfall data
        self.temp_moving_average() # sets self.avg_temp to an array of self.num_locations of temperature data


    def temp_sample (self,date):
        for month in self.climate:
            if (date >= month[0] and date < month[1]) or (date >= month[2] and date < month[3]):
                return rng.gumbel(month[4], 1, self.num_locations)

    def rain_sample (self,date):
        for month in self.climate:
            if (date >= month[0] and date < month[1]) or (date >= month[2] and date < month[3]):
                return rng.gumbel(month[5],.07, self.num_locations)

    def slide_window (self,window,sample_fn,*args):
        for i in range(self.k - 1):
            window[:,i] = window[:,i+1]
        window[:,-1] = sample_fn(*args)

    def update_rain (self,date):
        self.slide_window (self.rain_window,self.rain_sample,date)

    def update_temp (self,date):
        self.slide_window (self.temp_window,self.temp_sample,date)

    def  update_weather (self,date):
        self.update_rain (date)
        self.rain_moving_average()
        self.update_temp (date)
        self.temp_moving_average()

    def moving_average (self,window):
        return np.mean(window,axis=1)

    def rain_moving_average (self):
        self.avg_rain = self.moving_average(self.rain_window)

    def temp_moving_average (self):
        self.avg_temp = self.moving_average(self.temp_window)


def hist (vals,bins=15,density = True):
    count, bins, ignored = plt.hist(vals, bins, density=density)
    plt.show()

def dd (mean,std):
    hist(duration(mean,std),15)