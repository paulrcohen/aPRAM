#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:12:09 2020

@author: prcohen
"""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import aPRAM_utils as utils
from aPRAM_classes import Cohort, Column, Param, Counter

class Pop(SimpleNamespace):
    def __init__(self,size=None):
        self.size = size
        self.counters = []
        self.params = []
        self.columns = []
        self.static_cohorts = []
        self.dynamic_cohorts = []

    def make_pop_from_csv (self,file):
        """
        Reads a csv file and makes an aPRAM Column for each column in the
        file. This assumes that the first row in th csv file is the column
        headers. I'd like to be able to coerce columns into efficient numpy
        data types but numpy isn't letting me because some values are NaNs and
        numpy seems to require that arrays that include NaNs are float64.
        """
        data = pd.read_csv(file)
        for colname in data.columns:
            self.make_column(colname,np.array(data[colname]))


    def make_column (self,name,values):
        column = Column(name,values)
        self.__dict__.update({name:column})
        self.columns.append(column)

    def make_cohort (self, name, sexpr, dynamic = True):
        """ makes a cohort and registers it with self.static_cohorts or
        self.dynamic_cohorts depending on the 'dynamic' parameter."""
        cohort = Cohort(name,sexpr,dynamic)
        self.__dict__.update({name:cohort})
        if dynamic:
            self.dynamic_cohorts.append(cohort)
        else:
            self.static_cohorts.append(cohort)

    def make_param (self, name, initial_value, updater = None, *args):
        param = Param(name,initial_value,updater,*args)
        self.__dict__.update({name:param})
        self.params.append(param)

    def make_counter (self,name, initial_value = 0, increment = 1):
        counter = Counter(name, initial_value, increment)
        self.__dict__.update({name:counter})
        self.counters.append(counter)

    def reset (self):
        """ This deletes all the Params, counters, columns, and static and
        dynamic cohorts, and resets the lists that contain these entities to [].
        It preserves pop.size.  This would be used only during development and
        testing when one wants a 'clean' population.
        """
        size = self.size
        self.__dict__.clear()
        self.size = size
        self.counters, self.params, self.columns,self.static_cohorts, self.dynamic_cohorts = [],[],[],[],[]

    def describe (self,level=0):
        return (
            f"".join(
                [f"size: {self.size}\n",
                 f"Counters:\n",
                 *[f"{c.describe(level+1)}\n" for c in self.counters],
                 f"Params:\n",
                 *[f"{p.describe(level+1)}\n" for p in self.params],
                 f"Columns:\n",
                 *[f"{c.describe(level+1)}\n" for c in self.columns],
                 f"Static cohorts:\n",
                 *[f"{s.describe(level+1,print_selector=False)}\n" for s in self.static_cohorts],
                 f"Dynamic cohorts:\n",
                 *[f"{d.describe(level+1,print_selector=False)}\n" for d in self.dynamic_cohorts]
                ]
            ))


