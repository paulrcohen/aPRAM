#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:26:36 2020

@author: prcohen
"""

import numpy as np
from numpy.random import default_rng
rng = default_rng()

import pprint

from functools import partial
from inspect import signature
import aPRAM_utils as utils


#______________________________________________________________________________

class Cohort():
    """ A cohort is defined by a selector expression which must be a WN (when needed)
    expression (see aPRAM_expressions).  Cohorts may be dynamic or static.
    Dynamic cohorts are updated by re-evaluating the WN expression at each
    time step, whereas static cohort selectors are evaluated once at cohort creation time.

    It is a semantic error to use a Mod_Selector to define a Cohort.  Explain why!
    """

    def __init__(self, name, WN_obj, dynamic=True):
        self.name = name
        self.dynamic = dynamic
        self.selector = WN_obj
        self.selected =  self.select()   # select initial membership

    def select (self):
        if (not self.dynamic) and self.__dict__.get('selected') is not None:
            return self.selected

        else:
            if callable(self.selector):                 # e.g., a lambda expression
                self.selected = self.selector()
            elif hasattr(self.selector,'sexpr'):         #i.e., a WN sexpr
                self.selected = self.selector.eval()
            else:
                raise TypeError ("Param {self.name} updater must be callable or a WN expression")

            self.members  = np.where(self.selected)[0]  # indices of rows that satisfy the selector
            self.size = len(self.members)               # number of members
            return self.selected

    def get_members (self):
        return self.members

    def describe (self,level=0, print_selector = True):
        pp = pprint.PrettyPrinter(indent = 3 * level)
        indent0,indent1 = '   ' * level, '   ' * (level + 1)
        if self.size > 15:
            members = f"[{', '.join([str(i) for i in self.members[0:3]])} ... {', '.join([str(i) for i in self.members[-3:]])}]"
        else:
            members = f"{self.members}"
        if print_selector:
            selector_str = f'{indent1}selector: {pprint.pformat(self.selector)}\n{indent1}selected: {self.selected}'
        else:
            selector_str = ""
        return (
            f'{indent0}Cohort:\n'
            f'{indent1}name {self.name}\n'
            f'{indent1}size: {self.size}\n'
            f'{indent1}members: {members}'
            f'\n{selector_str}'
            )


#______________________________________________________________________________

class Column ():
    """ Early versions of aPRAM treated the population as a numpy two-dimensional
    array, but the overhead of operations on such a large structure increased
    runtimes, so now each column is its own object, and this also lets us define
    methods on columns."""

    def __init__(self,name,values):

        self.aPRAM_class = 'Column'
        self.name = name
        self.val = values

        """ Elementary comparison operators for columns. """
        self.eq = partial (self.op, fn = (lambda x,y : x == y) )
        self.ge = partial (self.op, fn = (lambda x,y : x >= y) )
        self.le = partial (self.op, fn = (lambda x,y : x <= y) )
        self.ne = partial (self.op, fn = (lambda x,y : x != y) )
        self.gt = partial (self.op, fn = (lambda x,y : x > y) )
        self.lt = partial (self.op, fn = (lambda x,y : x < y) )


    def op (self,y,fn):
        if callable (y):
            return fn(self.val,y.__call__())
        else:
            return fn(self.val,y)

    def assign (self,val,*args,**kwargs):
        selected = kwargs.get('selected')
        if selected is None:
            self.val[:] = val
        else:
            val_type= type(val)
            if val_type in [int,bool,float]:
                self.val[selected] = val
            elif val_type == np.ndarray:
                self.val[selected] = val[selected]
            else:
                raise ValueError (f"Trying to assign something other than int, bool, float, or numpy array to {self.name}")



    def get_val (self):
        return self.val

    def itself (self): return self

    def describe (self,level=0):
        indent0,indent1 = '   ' * level, '   ' * (level + 1)
        return (
            f'{indent0}Column:\n'
            f'{indent1}name {self.name}\n'
            f'{indent1}val: {self.val}'
            )


#______________________________________________________________________________


class Param ():
    """
    Whereas Columns must be arrays of length pop.size that represent
    agent attributes, Params can be anything: the transmission probability
    for a communicable disease, or a dict that maps age levels to severity of
    disease, or average rainfall, etc.

    Params require a name,  an initial value and an optional updater,
    which must be a WN expression.  If the update function refers to the
    parameter itself, then it must be wrapped in a lambda.

    Examples:
        p = Param('p', 2.1)
        # self.update returns 2.1

        q = Param('q', 1, updater = WN((lambda : q.val * 2)))
        # q.update() doubles its value.  However, the updater must be a lambda
        # because otherwise q refers to q.val in its own definition.
    """
    def __init__(self, name, val, updater=None):
        self.name = name
        self.val  = val
        self.updater = updater

    def update (self):
        if self.updater is not None:
            if callable(self.updater):
                self.val = self.updater()
            elif hasattr(self.updater,'sexpr'): # WN expression
                self.val = self.updater.eval()
            else:
                raise TypeError (f"Param {self.name} updater must be callable or a WN expression")

    def assign (self,val):
        self.val = val

    def get_val (self):
        return self.val

    def itself (self): return self

    def describe (self,level=0):
        indent0,indent1 = '   ' * level, '   ' * (level + 1)
        return (
            f'{indent0}Param:\n'
            f'{indent1}Name {self.name}\n'
            f'{indent1}Val: {self.val}'
            )

#______________________________________________________________________________

class Counter ():
    def __init__(self, name, initial_value=0, increment = 1):
        self.aPram_class = 'Counter'
        self.name = name
        self.val = initial_value
        self.increment = increment

    def advance (self):
        self.val += self.increment

    def assign (self, val):
        self.val = val

    def get_val (self): return self.val

    def describe (self,level=0):
        indent0,indent1 = '   ' * level, '   ' * (level + 1)
        return (
            f'{indent0}Counter:\n'
            f'{indent1}Name {self.name}\n'
            f'{indent1}Val: {self.val}'
            )
#______________________________________________________________________________


class Mod_Selector ():
    """
    A Mod can have k alternative mod_lists. When k > 1 then a Mod_Selector is
    used to assign one mod_list to each agent in a cohort. A Mod_Selector
    returns a multinomial distribution self.selected over the integers
    0 ... k-1 with probabilities p0...pk-1.  The argument prob_spec must be a
    list of k probabilities or a sexpr that returns a list of k probabilities.

    A Mod_Selector needs a cohort argument because 1) the size of self.selected
    must equal the cohort size, 2) the select method needs to know whether the
    cohort size has changed.
    """

    def __init__(self, cohort, prob_spec, k, report=False):

        from aPRAM_settings import pop # need this because we need pop.size
        self.aPram_class = 'Mod_Selector'

        self.cohort = cohort
        self.prob_spec = prob_spec
        self.k  = k
        self.n0 = cohort.size # n0 is the last value of cohort size
        self.n1 = cohort.size # n1 is the current value of cohort_size
        self.report = report

        """ prob_spec must be either a list of probabilities, a Param whose
        value is a list of probabilities, or a callable -- a method or lambda or
        function -- that returns a list of probabilities. """

        if hasattr(self.prob_spec,'sexpr'):  # it's a WN expression
            self.prob_fn = (lambda : self.prob_spec.eval())
            self.probs = utils.check_probs(self.prob_fn())

        elif callable(self.prob_spec):
            self.prob_fn = prob_spec
            self.probs = utils.check_probs(self.prob_fn())

        else:
            self.prob_fn = None
            self.probs = utils.check_probs(prob_spec)



        """ Get the initial self.selected. At this point, the cohort size could
        be zero, or very small, which will cause problems if later we try to resample
        from self.selected. Furthermore, the initial cohort size might be tiny, which
        again would cause infidelity to probabilities when we resample.  A possibly
        wasteful solution is to have self.selected_initial be the size of the population
        and to resample from it.
        """

        self.selected_initial = rng.choice(
            self.k,           # integers from 0...k-1
            p = self.probs,   # k probabilities
            size = pop.size,
            replace=True).astype(np.int8)

        self.selected = rng.choice(self.selected_initial, size = self.n1, replace = True)

        if self.report: print (f"In Mod_Selector: probs: {self.probs}, self.selected: {self.selected[:20]}...\n ")

    def select (self):
        """ Rebuilding a multinomial distribution self.selected is the most
        expensive operation in aPRAM, so it's worth being careful about when to do it:

            • If self.probs is static, meaning it is defined as a list of
            probabilities when the Mod_Selector is defined, then we only build
            self.selected once. We might choose to roll self.selected before
            selection to ensure that agents can have different choices on
            subsequent invocations of a Mod.

            • If self.probs is static but the cohort size changes to n, and
            self.selected exists, then we can resample with replacement to get a
            sample of n from self.selected.  This is much cheaper than rebuilding
            self.selected when n is relatively small, but is cheaper even when
            resampling a sample of the size of the original self.selected.

            • Rebuilding self.selected is necessary only when self.probs is dynamic,
            that is, a callable that needs to be updated at each time step.

        Defaults:  If self.probs is static, build self.selected and 1) if the
        cohort size changes, resample from self.selected, otherwise 2) roll self.selected.
        If self.probs is dynamic, rebuild self.selected.
        """
        self.n1 = self.cohort.size  # get current cohort size

        if self.prob_fn is None: # probabilities are static, so roll or resample M
            if self.report: print(f"Static probs {self.probs} n0 = {self.n0}, n1 = {self.n1}")

            if self.n0 == self.n1: # last cohort size equals current cohort size
                self.roll() # no need to resample from M, simply roll it
                if self.report: print ("Rolling self.selected\n")

            else:
                self.selected = rng.choice(self.selected_initial, size = self.n1, replace = True)
                # construct a new self.selected for new sample size by resampling from the previous one
                if self.report: print ("Resampling self.selected\n")

        else:                    # probabilities are dynamic
            self.probs = utils.check_probs(self.prob_fn())  # check that the changed probs are legit
            #old: self.probs = utils.check_probs(ap.ev(self.prob_fn))  # check that the changed probs are legit
            self.selected = rng.choice(self.k, p = self.probs, size = self.n1, replace=True).astype(np.int8)
            # int8 is much faster and we won't have > 256 modifications!
            if self.report: print (f"Rebuilding self.selected, probs = {self.probs}\n")

        # set n0 to n1 so we can track whether sample size changes next time
        self.n0 = self.n1

    def roll (self):

        """ Instead of reconstructing the selector with self.select it is much
        faster to roll the selector, ensuring that each agent gets a new mod_list
        on each time step, but not changing the selector itself."""

        self.selected = np.roll(self.selected, shift=rng.integers(10))

    def describe (self,level=0):
        pp = pprint.PrettyPrinter(indent = 3 * level)
        indent0 = '   ' * level
        indent1 = '   ' * (level + 1)
        return (
            f'{indent0}Mod_Selector:\n'
            f'{indent1}cohort: {self.cohort.name}\n'
            f'{indent1}prob_spec: {pprint.pformat(self.prob_spec)}\n'
            f'{indent1}prob_fn: {pprint.pformat(self.prob_fn)}\n'
            f'{indent1}probs: {self.probs}'
            )


#______________________________________________________________________________

class Mod ():
    """
    A Mod represents one or more modifications to a cohort. A Mod requires:

            1) The cohort to which modifications apply
            2) A list of k ≥ 1 modlists
            3) If k > 1, a prob_spec, which may be a Mod_Selector or the
            elements from which to build a Mod_Selector

    1) A cohort object is passed as an argument. It may be a static or dynamic
    cohort, but if it is dynamic then the updates have already happened before
    the mods happen.

    2) Each modlist is a list of zero or more acts.

    3) If there's only one mod_list (i.e., k ==1) , then it applies to all cohort
    members.  In this case, prob_spec need not be specified (i.e., defaults
    to None). If k > 1 then the Mod requires a Mod_Selector.  There are three
    options:

        a. if prob_spec.__class_ == Mod_Selector, then prob_spec is
        used as the Mod_Selector,

        b. prob_spec is a list of probabilities

        c. prob_spec is a lambda expression that returns a list of probabilities.

    In cases b or c, Mod will build a Mod_Selector from these elements.

    Assume r is a Mod_Selector object.  r.selected is a multinomial distribution
    over the integers 0 to k - 1.  The ith mod_list is applied to cohort.members[r.selected==i].

    The Mod_Selector may be static or dynamic, and in either case might be rolled.
    Dynamic selectors might be rolled, resampled or rebuilt.  See Mod_Selector.
    """

    def __init__(self, name, cohort, mods, prob_spec = None):
        self.name = name
        self.cohort = cohort

        # syntactic check of mods:
        if not (type(mods) == list and type(mods[0]) == list):
                raise ValueError(f"{mods} is not a list of mod_lists as required.")

        if len(mods) > 1 and prob_spec is None:
            raise AttributeError("Need a mods_selector when mods contains more than one action list.")

        # parse the mods catching empty modlists
        self.mods = [[None] if modlist == [] else modlist for modlist in mods]

        if len(mods) == 1 :  # there's no need for a Mod_Selector or prob_spec
            self.mod_selector = None

        else:
            if prob_spec.__class__ == Mod_Selector:    # if a Mod_Selector is passed in
                self.mod_selector = prob_spec
            else:                                      # else need to make one
                self.mod_selector = Mod_Selector (self.cohort, prob_spec, len(mods))


    def select (self):
        """ This is shorthand for mod.mod_selector.select().  Not all Mods have
        Mod_Selectors (only those Mods that have multiple mod_lists do), so this
        function checks whether this Mod has a selector."""

        if self.__dict__.get('mod_selector') is not None:
                self.mod_selector.select()


    def do_mods (self):

        def eval_mod(mod,selected):
            if callable(mod):
                return mod(selected=selected)
            elif hasattr(mod,'sexpr'):
                return mod.eval(selected=selected)
            else:
                raise TypeError (f"Param {self.name} updater must be callable or a WN expression")

        if len(self.mods) == 1:         # Only one modlist, so it applies to entire cohort
            selected = self.cohort.members
            for mod in self.mods[0]:
                if mod is not None: eval_mod(mod,selected)
        else:
            i = -1
            for mod_list in self.mods:
                i += 1
                selected = self.cohort.members [ self.mod_selector.selected == i ]
                for mod in mod_list:
                    if mod is not None: eval_mod(mod,selected)


    def describe (self,level=0):
        indent0, indent1 = '   ' * level, '   ' * (level + 1)
        pp = pprint.PrettyPrinter(indent = 3 * level)
        modstr = f"{indent1}".join(
            [f"{pprint.pformat(mod, indent = level + 3)}\n" for mod in self.mods])
        if self.mod_selector is not None:
            mod_selector_str = self.mod_selector.describe(level+1)
        else:
            mod_selector_str = None

        return (
            f'{indent0}Mod\n'
            f'{indent1}name: {self.name}\n'
            f'{indent1}cohort: {self.cohort.name}\n'
            f"{indent1}mods: {modstr}"
            f'{indent1}mod_selector: {mod_selector_str}\n'
            )



class Act ():
    """
    Acts are arbitrary chunks of code that run at setup or in the main
    simulation loop.  Whereas Columns and Params hold simulation state
    and are updated by Mod ColMods and Param update_fns, respectively,
    Acts are not intended to hold any state information but simply make
    things -- particularly non-sim processes -- happen.  Acts are also
    useful for printing diagnostic information when debugging sims.

    Acts require an action0 argument and optionally a condition and action1
    argument, providing rudimentary if-then-else control.

    """
    def __init__(self, action0, condition=None, action1 = None):
        self.condition = self.make_callable(condition)
        self.action0 = self.make_callable(action0)
        self.action1 = self.make_callable(action1)

    def make_callable (self,expr):
        if expr is None:
            return None
        elif callable (expr):
            return expr
        elif hasattr(expr,'sexpr'):
            return expr.eval
        else:
            raise TypeError ("Act {self.name} condition/action must be callable or a WN expression")

    def do_act (self):
        if self.condition is None:
            self.action0()
        else:
            if self.condition():
                self.action0()
            else:
                if self.action1 is not None:
                    self.action1()






#%%

##################################################################
########
########   TO DO
########
##################################################################

# No provision for having a mod apply to multiple cohorts

