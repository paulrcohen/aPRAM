import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator as op
from numpy.random import default_rng
rng = default_rng() # stuff for random sampling

import aPRAM_utils as utils
from aPRAM_expressions import WN
from aPRAM_settings import pop,sim
from rice_model_utils import hist, Weather


#______________________________________________________________________________
pop.size = 100000
pop.reset ()
sim.reset ()

#______________________________________________________________________________

def duration (mean,std,n=pop.size):
    s = rng.poisson(std**2,pop.size)
    s += mean-std**2
    return s
#______________________________________________________________________________


# The Weather will set the temperature and rainfall for each farm
# on each day and maintain moving averages of temp and rainfall
w = Weather(num_locations = pop.size)

# date will advance by 1 daily
pop.make_counter('date')

pop.make_param(name = 'Predicted_Y1_tonnage',
                   initial_value = 0,
                   updater = (lambda : int(np.sum(pop.yield_1.val * pop.hct.val))))

pop.make_column ('avg_rain', w.avg_rain)
pop.make_column ('avg_temp', w.avg_temp)


# distribution of planting dates for first and second planting; FAO/GIEWS
pop.make_column('planting_date_1', duration(160,3))
pop.make_column('planting_date_2', duration(315,3))

# distribution of planting dates for first and second harvesting
# these numbers will shift depending on simulated weather conditions
pop.make_column('harvest_date_1', duration(270,3))
pop.make_column('harvest_date_2', duration(425,3))

# make the average farm size 1 hct
pop.make_column('hct', rng.lognormal(.1,.25,pop.size))

# yield per hectare, croppings 1 and 2. These will
# change -- generally downward toward the actual mean
# of 4 tons/hct -- during the sim as conditions change
pop.make_column('yield_1', rng.gumbel(6,3,pop.size))
pop.make_column('yield_2', rng.gumbel(6,3,pop.size))


# Some farms are in low-lying areas prone to flooding
pop.make_column('flood_threshold', rng.gumbel(.25,.07, pop.size))


# Cohorts
pop.make_cohort('flooded', WN(op.gt, pop.avg_rain, pop.flood_threshold))
pop.make_cohort('chilled', WN(op.lt, pop.avg_temp, 4.75))
pop.make_cohort('two_croppings', WN(op.lt, (op.add, pop.harvest_date_1.val, 30), pop.planting_date_2))

# We also will need a "population cohort" whose members are all the agents in the population:
pop.make_cohort('P', (lambda: np.ones(pop.size).astype(bool)), dynamic=False)


# Dynamics

sim.make_act (
    name = 'update_weather',
    fn = (lambda : w.update_weather(pop.date.val)),
    sim_phase = 'loop'
    )


sim.make_mod(
    name = 'local_weather',
    cohort = pop.P,
    mods = [[WN(pop.avg_rain.assign, (lambda : w.avg_rain)),
             WN(pop.avg_temp.assign, (lambda : w.avg_temp))]],
    sim_phase = 'loop'
    )


sim.make_mod(
    name = 'flood_damage',
    cohort = pop.flooded,
    mods = [
        [ WN (pop.yield_1.assign, (op.mul, pop.yield_1.val, .9)),
          WN (pop.harvest_date_1.assign, (op.add, pop.harvest_date_1, 5)) ]
        ],
    sim_phase = 'loop'
    )

sim.make_mod(
    name = 'cold_damage',
    cohort = pop.chilled,
    mods = [
        [ WN (pop.yield_1.assign, (op.mul, pop.yield_1.val, .7)),
          WN (pop.harvest_date_1.assign, (op.add, pop.harvest_date_1, 5)) ],
        []
        ],
    prob_spec = (.1,.9),
    sim_phase = 'loop'
    )


# sim.make_act (
#     name = 'diagnostics',
#     fn = (lambda : print (f"date: {pop.date.val}, w.avg_rain: {w.avg_rain[:3]}    pop.avg_rain: {pop.avg_rain.val[:3]}")),
#     sim_phase = 'loop'
#     )


# record data with this function
def probe_fn (day):
    record = [day, pop.Predicted_Y1_tonnage.val]
    record.extend([c.size for c in [pop.flooded,pop.chilled,pop.two_croppings]])
    return record

probe_labels = ['day','tonnage','flooded','chilled','two_croppings']

sim.num_iterations  = 400
sim.probe_labels    = probe_labels
sim.probe_fn        = probe_fn


# tell sim to run
sim.run_simulation()


df = pd.DataFrame(sim.records,columns=probe_labels)
A = ['tonnage','flooded','chilled','two_croppings']
B = ['tonnage']
df.loc[:,A].plot(secondary_y=B) #mark_right=False,
plt.show()
