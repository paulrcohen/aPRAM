#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:30:59 2020

@author: prcohen
"""
import numpy as np
import time

def desc (obj,attributes,level):
    indent = '   ' * level
    return f"".join(
        [f"\n{indent}{a}: {obj.__dict__.get(a)}\n" for a in attributes])

def describe (obj, attrs = None, level = 0, max_level=3):
    import aPRAM_population, aPRAM_sim, aPRAM_classes
    classes_to_describe = [aPRAM_population.Pop, aPRAM_classes.Parameter,
                           aPRAM_classes.Counter,aPRAM_classes.Column,
                           aPRAM_classes.Mod,aPRAM_classes.Mod_Selector,
                           aPRAM_sim.Sim]

    def describe_helper (attr,val,level):
        indent = '     ' * level
        if level < max_level:
            #print (f"val.__class__ {val.__class__}")
            if val.__class__ in classes_to_describe:
                return f"{indent}{attr}: {val}\n{describe(val, level = level+1, max_level=max_level)}"
            else:
                return f"{indent}{attr}: {val}\n\n"
        else:
            return f"{indent}{attr}: {val}\n\n"

    indent = '     ' * level

    if attrs is None:
        d = obj.__dict__
    else:
        d = {k:obj.__dict__[k] for k in attrs}

    return f"".join(
        [describe_helper(attr,val,level) for attr,val in d.items()]
        )

def is_number(x):
    return isinstance(x, (int, float, np.float64))

# print(is_number(1))
# print(is_number(1.0))
# print(is_number([1]))
# print(is_number(-1))
# print(is_number(np.random.randint(0,5)))
# print(is_number(np.random.randint(0,5,2)))

def count (boolean_array):
    try:
        return np.sum(boolean_array)
    except:
        raise ValueError("The count function requires a boolean array argument.")

def distribution (arr):
    u,c = np.unique(arr,return_counts = True)
    print(f"Value\tFrequency")
    print(f"=====\t=========")
    for v,f in zip(u,c):
        print(f"{v}\t{f}")
    print()


def check_probs (probs, k = None, fix_limited_precision = True):
    """Given a list or tuple of numbers, this checks whether they can be probabilities
    and whether there are k of them. If so, it returns them, otherwise it raises
    ValueErrors.  If fix_limited_precision = True, then the probabilities are
    rounded at the 12th decimal place. This ensures that, say, .3999999999999 is
    represented as .4."""

    if type(probs) in [list,tuple]:
        if k is None or len(probs) == k:
            if all ([is_number(p) and p >= 0 and p <= 1 for p in probs] ):
                if fix_limited_precision:
                    s = round(sum(probs),12)
                else:
                    s = sum(probs)
                if s == 1:
                    return probs
                else:
                    raise ValueError(f"Probabilities {probs} do not sum to 1.0")
            else:
                raise ValueError(f"Some of {probs} aren't probabilities.")
        else:
            raise ValueError(f"{probs} should contain {k} probabilities")
    else:
        raise ValueError(f"{probs} is not a list/tuple")


# print(check_probs((1,)))
# print(check_probs((.1,.2,.3)))
# print(check_probs((.1,.2,.3,.4)))
# print(check_probs((.1,.2,.3,.39999999999999)))
# print(check_probs((1,2)))
# print(check_probs((1,.1)))
# print(check_probs((.1,.1)))
# print(check_probs((1,'a')))
# print(check_probs((1,),3))



class Timer ():
    def __init__(self):
        self.times = [time.time()]
        self.annotations = []

    def event (self,annotation):
        self.annotations.append(annotation)
        self.times.append(time.time())

    def report (self):
        print("\n\n-------------------")
        intervals = [x - y for x,y in zip(self.times[1:],self.times[0:-1])]
        for a,t in zip(self.annotations,intervals):
            print(a,round(t,5))
        if self.annotations == []: # no events have been recorded
            total_time = time.time() - self.times[0]
        else:
            total_time = self.times[-1]-self.times[0]
        print("\nTotal time = {:.5f}".format(total_time))
        print("-------------------\n\n")
