#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:50:47 2020

@author: prcohen
"""


from aPRAM_settings import pop
from aPRAM_classes import Mod,Act

class Sim ():

    def __init__(self, dynamic_cohorts = [], static_cohorts = [],
                 disable_mods = [], num_iterations = 0,
                 probe_fn = None, probe_labels = None
                 ) :
        self.dynamic_cohorts = dynamic_cohorts # if None, we'll get it from pop
        self.static_cohorts  = static_cohorts
        self.setup_mods      = []
        self.loop_mods       = []
        self.setup_acts      = []
        self.loop_acts       = []
        self.num_iterations  = num_iterations
        self.probe_fn        = probe_fn
        self.probe_labels    = probe_labels
        self.records         = []

    def make_mod (self, name, cohort, mods, prob_spec = None, sim_phase = None):
        """ This creates a Mod object and registers it with the simulator """
        mod = Mod(name, cohort, mods, prob_spec)
        self.__dict__.update({name:mod})
        if sim_phase == 'loop':
            self.loop_mods.append(mod)
        elif sim_phase == 'setup':
            self.setup_mods.append(mod)
        else:
            raise AttributeError ("sim_phase must be 'loop' or 'setup'")

    def make_act (self, name, action, condition = None, alt_action = None, sim_phase = None):
        """ This creates an Act object and registers it with the simulator """
        act = Act(action,condition,alt_action)
        self.__dict__.update({name:act})
        if sim_phase == 'loop':
            self.loop_acts.append(act)
        elif sim_phase == 'setup':
            self.setup_acts.append(act)
        else:
            raise AttributeError ("sim_phase must be 'loop' or 'setup'")

    def inner_loop (self, acts, mods, *cohorts):
        # Acts run first, then Mod_Selectors, then Mods, then Cohorts are updated
        for act in acts: act.do_act()
        for mod in mods: mod.select()
        for mod in mods: mod.do_mods()
        for cohort in cohorts: cohort.select()

    def run_setup (self):
        self.inner_loop (self.setup_acts, self.setup_mods, *self.static_cohorts, *self.dynamic_cohorts)

    def one_iteration (self):

        # Update all counter and parameter values
        for counter in pop.counters: counter.advance()
        for param in pop.params: param.update()

        # run all the Acts and Mods
        self.inner_loop (self.loop_acts, self.loop_mods, *self.dynamic_cohorts)


    def run_simulation (self, report=False):
        """ If no cohorts were specified as parameters, then we get them from pop. """

        if self.dynamic_cohorts == []: self.dynamic_cohorts = pop.dynamic_cohorts

        if self.static_cohorts  == []: self.static_cohorts = pop.static_cohorts

        self.run_setup()

        for i in range(self.num_iterations):
            self.one_iteration()
            self.records.append(self.probe_fn(i))

    def reset (self):
        """ This deletes all the Cohorts and Mods, and resets the lists that
        contain these entities to []. It preserves the probe_fn and probe_labels.
        This would be used only during development and testing when one wants
        a 'clean' sim.
        """
        fresh_dict = {
            'num_iterations' : None,
            'dynamic_cohorts': [],
            'static_cohorts' : [],
            'setup_mods' : [],
            'loop_mods' : [],
            'setup_acts' : [],
            'loop_acts' : [],
            'probe_fn' : self.probe_fn,
            'probe_labels' : self.probe_labels,
            'records' : []}
        self.__dict__.clear()
        self.__dict__.update(fresh_dict)


    def describe (self,level=0):
        return (
            f"".join(
                [f"Number of iterations: {self.num_iterations}\n",
                 f"Records: \n{self.records}\n",
                 f"Static cohorts:\n",
                 *[f"{s.describe(level+1,print_selector=False)}\n" for s in self.static_cohorts],
                 f"Dynamic cohorts:\n",
                 *[f"{d.describe(level+1,print_selector=False)}\n" for d in self.dynamic_cohorts],
                 f"Setup Mods:\n",
                 *[f"{s.describe(level+1)}\n" for s in self.setup_mods],
                 f"Loop Mods:\n",
                 *[f"{d.describe(level+1)}\n" for d in self.loop_mods],
                 f"Setup Acts:\n",
                 *[f"{s.describe(level+1)}\n" for s in self.setup_acts],
                 f"Loop Acts:\n",
                 *[f"{d.describe(level+1)}\n" for d in self.loop_acts]
                ]
            ))
