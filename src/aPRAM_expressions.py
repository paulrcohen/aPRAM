#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 18:12:59 2020

@author: prcohen
"""

import operator
import aPRAM_classes
from aPRAM_classes import Cohort, Column, Mod, Mod_Selector, Param, Counter

"""
The main classes in aPRAM -- Cohorts, Columns, Parameters, Mods and
Mod_Selectors -- are dynamic, meaning that their values can change during
a simulation.  Often, we want to specify computations that will happen in
a future context; for example:

    • the criteria for Cohort membership should be evaluated when needed,
    not necessarily when the Cohort is defined.

    • Mods to Columns such as multiplying their values by .95 should
    be done when needed -- using whichever values the Column has at the time --
    not necessarily when the Mod is defined.

    • Probabilities of different mods should be evaluated when needed, not
    necessarily when the Mod or probability functions are defined.

One way to achieve when-needed computation is to hold apart functions and
their arguments, combining them only as needed. The WN class has methods to
parse expressions that are required by the definitions of aPRAM objects and
evaluate them only when needed.

WN expressions have a Lisp-like syntax:  If the first element in a list is
callable, then the remaining elements are taken as arguments.  For example,

wn = WM.parse(pop.col.assign, (rng.integers, 0, (foo, pop.p, 12), 30))

when evaluated will cause aPRAM to assign col values obtained by generating 30
random integers between 0 and a number that depends on the function foo applied
to the parameter pop.p and 12.  Nothing is done to col until wm is evaluated. Keyword
arguments can be supplied, for example,

wn.eval(selected=C.members)

tells aPRAM to evaluate the previously parsed expression but apply the results
only to the members of Cohort C.
"""

class WN ():
    def __init__(self, *expr):
        if len(expr) == 1:          # sometimes expr is a naked callable or a number or parameter name etc.
            self.sexpr = expr[0]
        else:
            self.sexpr = self.parse(expr)

    def parse (self,expr):
        collect = []
        if type(expr) not in (list,tuple):
            if callable(expr):
                collect = [expr]
            else:
                collect = expr
        elif callable(expr[0]):
            collect = [expr[0]]
            collect.extend([self.parse(e) for e in expr[1:]])
        else:
            collect.extend([self.parse(e) for e in expr])
        return collect


    def eval (self, *args, report=False, **kwargs):
        return self.ev(self.sexpr, *args, report=report, **kwargs)


    def ev (self, expr, *args, report=False , **kwargs):

        def r (f):
            if report: print(f)

        r (f"\nentering ev with expr: {expr}, class = {expr.__class__}")
        if type(expr) != list:
            if callable(expr):
                r (f"\nnaked callable: {expr}")
                r (f"calling it returns {expr()}")
                return expr(*args)

            elif expr.__class__ in [aPRAM_classes.Column,aPRAM_classes.Param,aPRAM_classes.Counter]:
                r (f"\nreturning {expr.val}\n")
                return expr.val

            else:
                r (f"\nKickout returning {expr}")
                return expr




        elif callable(expr[0]):
            r (f"\ncallable {expr[0]}\n args : {expr[1:]}")
            return expr[0](*[self.ev(e,*args,report=report) for e in expr[1:]],*args,**kwargs)

        else:
            return expr

# # #%%
# # ####################  Testing ##########################

# import numpy as np
# from numpy.random import default_rng
# rng = default_rng() # stuff for random sampling
# from aPRAM_settings import pop
# pop.size = 20
# pop.make_column('h',np.zeros(pop.size))

# #%%  Params

# ## Params have optional updaters.  Without one, a Param simply holds a value:
# pop.make_param('p',2.1)
# print(pop.p.val)
# print(pop.p.get_val())


# ## We can add an updater after defining the param:

# pop.p.updater = WN(operator.mul, pop.p, 2)
# pop.p.update()
# print (pop.p.val) # 4.2


## We can define the updater in the param definition.
# ## This says pop.q will update to have twice the value of pop.p
# pop.make_param('q',2, WN(operator.mul, pop.p, 2))
# print(pop.q.val)
# pop.q.update()
# print(pop.q.val)

# ## Now change pop.p.val and see that nothing happens to pop.q.val until it's updated
# pop.p.assign(10)
# print (pop.q.val)  # value hasn't changed
# pop.q.update()
# print (pop.q.val)  # now the value is twice the new value of pop.p

# ## But what if we want to define an updater as part of the parameter definition
# ## AND we want the updater to refer to the parameter itself? Like this:
# ## pop.make_param('s',1,WN((operator.mul, pop.s, 2))).  That's a problem
# ## because we can't define an updater to do something to pop.s when pop.s isn't
# ## defined yet.  The only ways to do it are as above, defining the parameter and
# ## then defining its updater, or wrapping the updater in a lambda:

# pop.make_param('s',1, WN(lambda : pop.s.val * 2))
# print(pop.s.val) # 1
# pop.s.update()
# print(pop.s.val) # 2

# ## Finally, note that we must refer to pop.s.val, not just pop.s and we
# ## can't use prefix notation, whereas if we defined pop.s first and then
# ## defined it's updater we could say:

# pop.make_param('s',1)
# pop.s.updater = WN(operator.mul, pop.s, 3)
# pop.s.update()
# print(pop.s.val) # 3



# #%%
# # ### Assignment to Column values

# pop.h.val = np.zeros(pop.size)
# print(pop.h.val)

# ## Columns don't have updaters, like parameters, but they are modified by Mods.
# ## Mods are made up of lists of acts.  This act says assign column pop.h the
# ## an array of random integers generated by the Numpy function rng.integers.
# ## Note that the call to that function is in WN form, namely, a function
# ## name followed by its arguments.

# act = WN(pop.h.assign, (rng.integers, 0, 10, pop.size))
# act.eval()
# print(pop.h.val)

# ## One gets the same result by wrapping the call to rng.integers in a lambda
# act = WN(pop.h.assign, (lambda : rng.integers(0, 10, pop.size)))
# act.eval()
# print(pop.h.val)

# ## We can also pass in variable-length argument lists
# act = WN(pop.h.assign, (lambda *args: rng.integers(args[0], args[1], pop.size)))
# act.eval(0,10)
# print(pop.h.val)

# ## To do: act = WN((pop.h.assign, (rng.integers, args[0], args[1], pop.size))) doesn't work

# ## Suppose we want to use the WN form instead of lambdas, but we want to pass in values:
# pop.make_param ('p', 50)
# act = WN(pop.h.assign, (rng.integers, 0, pop.p, pop.size))
# act.eval()
# print (pop.h.val)

# ## now if the value of pop.p changes, evaluating act gives a different result:
# pop.p.val = 2
# act.eval()
# print (pop.h.val)

# ## An important aspect of aPRAM is doing things to only selected rows, not all rows
# ## We use the keyword argument 'selected' for this:
# pop.h.val = np.zeros(pop.size)
# print(pop.h.val)

# act = WN(pop.h.assign, (rng.integers, 0, 10, pop.size))
# act.eval(selected=np.array([0,1,2,3,4]))
# print (pop.h.val) # only the first five numbers have changed

# ## We can change which rows are selected and re-evaluate:
# act.eval(selected=np.array([10,11,12]))
# print (pop.h.val)
# # the first five rows were not selected so their values don't change, but three later rows do

# ## Once a column is defined, we can refer to it in any act:
# print (pop.h.val)

# act = WN(pop.h.assign, (operator.iadd, pop.h, 10))
# act.eval()
# print (pop.h.val)

# ###### Cohorts ########
# ## Cohorts are defined by boolean selection criteria.  Because agents can meet
# ## a cohort's criteria at one time and not another (i.e., cohort membership is
# ## dynamic), the criteria should be specified as WN expressions:

# pop.h.val = rng.integers(0,9,pop.size)

# ## This defines a cohort c with criterion that pop.h ≤ 4:
# c = Cohort('c', WN(pop.h.le,4))
# c.select()

# print(pop.h.val)
# print(c.selected.astype(int)) #True and False are written as 1 and 0, respectively
# print(c.members)

# # If h.val changes, so will cohort membership
# pop.h.val = rng.integers(0,9,pop.size)
# c.select()
# print(pop.h.val)
# print(c.selected.astype(int)) #True and False are written as 1 and 0, respectively
# print(c.members)

# ##### Miscellaneous ######

# ## WN expressions can contain functions
# def foo (x): return 2 * x
# def bar (): return pop.p.val

# pop.h.val = rng.integers(0,9,pop.size)
# print (pop.h.val)
# pop.p.val = 5

# act = WN(pop.h.assign, (foo,bar))
# act.eval()
# print (pop.h.val)

# pop.p.val = 3
# act.eval()
# print (pop.h.val)

# ## WN expressions simply 'pass through' anything that doesn't have the form (callable, *args):

# pop.p.val = [.2,.8]
# x = WN(pop.p)
# print(x.eval()) # [.2,.8]

# x = WN([.3,.2])
# print(x.eval()) # [.3,.2]

# x = WN(True)
# print(x.eval()) # True

# x = WN(None)
# print(x.eval()) # None

# x = WN((None))
# print(x.eval()) # None

# x = WN(np.pi)
# print(x.eval()) # 3.1415..

# x = WN(pop.h)
# print(x.eval())

# def foo (): return 27
# x = WN(foo)
# print(x.eval()) # 27
