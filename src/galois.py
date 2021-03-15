import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng() # stuff for random sampling

import aPRAM_utils as utils
from aPRAM_expressions import WN

"""
aPRAM_settings creates a population called pop and a simulator called sim.  Both are
"empty" but will be "filled in" as Cohorts and Columns and Mods etc. are defined.
However, all these classes need to know the population size, so this is the first
thing to set after pop and sim have been imported.
"""

from aPRAM_settings import pop,sim

#______________________________________________________________________________
pop.size = 1000

pop.reset ()
sim.reset ()

pop.make_column("health", np.zeros(pop.size))
S = 1
I = 2
R = 3

pop.make_cohort("Population", lambda: np.ones(pop.size).astype(bool))
pop.make_cohort("S", lambda: pop.health.eq(S))
pop.make_cohort("I", lambda: pop.health.eq(I))
pop.make_cohort("R", lambda: pop.health.eq(R))

sim.make_mod(
    name="Initialize",
    cohort=pop.Population,
    mods=[
        [lambda selected : pop.health.assign(S,selected=selected)],
        [lambda selected : pop.health.assign(I,selected=selected)],
        [lambda selected : pop.health.assign(R,selected=selected)],
    ],
    prob_spec= lambda : [
        (997 / 1000.0),
        (3 / 1000.0),
        (0 / 1000.0),
    ],
    sim_phase="setup",
)

# # manually run the setup mod
# m0 = sim.setup_mods[0]  # get the mod from the sim
# m0.select()             # select who will get which modlist
# m0.do_mods()            # execute it

# print(np.unique(pop.health.val, return_counts=True))

# #  update the cohorts (otherwise you get the divide-by-zero error)
# pop.S.select()          
# pop.I.select()
# pop.R.select()


sim.make_mod(
    name="Infect",
    cohort=pop.S,
    mods=[
        [lambda selected: pop.health.assign(I,selected=selected)],
        [],
    ],
    prob_spec=lambda: [
        ((((0.4 * pop.S.size) * pop.I.size) / ((pop.S.size + pop.I.size) + pop.R.size)) / ((((0.4 * pop.S.size) * pop.I.size) / ((pop.S.size + pop.I.size) + pop.R.size)) + ((4.0e-2 * pop.I.size) + 0.0))),
        (1.0 - ((((0.4 * pop.S.size) * pop.I.size) / ((pop.S.size + pop.I.size) + pop.R.size)) / ((((0.4 * pop.S.size) * pop.I.size) / ((pop.S.size + pop.I.size) + pop.R.size)) + ((4.0e-2 * pop.I.size) + 0.0)))),
    ],
    sim_phase="loop",
)


sim.make_mod(
    name="Remove",
    cohort=pop.I,
    mods=[
        [lambda selected: pop.health.assign(R, selected=selected)],
        [],
    ],
    prob_spec=lambda: [
        ((4.0e-2 * pop.I.size) / ((((0.4 * pop.S.size) * pop.I.size) / ((pop.S.size + pop.I.size) + pop.R.size)) + ((4.0e-2 * pop.I.size) + 0.0))),
        (1.0 - ((4.0e-2 * pop.I.size) / ((((0.4 * pop.S.size) * pop.I.size) / ((pop.S.size + pop.I.size) + pop.R.size)) + ((4.0e-2 * pop.I.size) + 0.0)))),
    ],
    sim_phase="loop",
)


# m1 = sim.loop_mods[0]  # get the mod from the sim
# m2 = sim.loop_mods[1]  # get the mod from the sim

# # now loop over the loop mods five times

# for i in range (5):
    
#     m1.select()             # select who will get which modlist
#     m1.do_mods()            # execute it


#     m2.select()             # select who will get which modlist
#     m2.do_mods()            # execute it

#     print(np.unique(pop.health.val, return_counts=True))
    
#     pop.S.select()          #  update the cohorts
#     pop.I.select()
#     pop.R.select()


def probe_fn (day):
    record = [day, sim.Infect.mod_selector.probs[0]]
    record.extend([c.size for c in [pop.S,pop.I,pop.R]])
    return record

probe_labels = ['day','p','S','I','R']

sim.num_iterations  = 10
sim.probe_labels    = probe_labels
sim.probe_fn        = probe_fn
#%%


# tell sim to run
sim.run_simulation()

# store sim.records -- the data -- in a pandas dataframe
df1 = pd.DataFrame(sim.records,columns=probe_labels)

#plot the results
ax = df1['I'].plot(label = 'Num. infected')
df1['S'].plot(ax=ax, label = 'Num. susceptible')
ax.legend()
plt.title(label = 'Epidemic dynamics over '+str(sim.num_iterations)+' days.')
plt.show()