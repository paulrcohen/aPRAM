#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:39:20 2020

@author: prcohen
"""

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
pop.size = 100000

pop.reset ()
sim.reset ()

#_____________________________________________________________________________
# The only attributes we'll need are health and quarantine status

pop.make_column('health',np.zeros(pop.size))
pop.make_column('quarantined', np.zeros(pop.size))

#______________________________________________________________________________
# define simulation parameters and functions
pop.make_param('beta', .3)
pop.make_param ('p_quarantine',[.2,.8])

def p_transmission (beta):
    # Infectious agents -- those in cohort F -- can transmit infections
    # The probability of being an infectious agent is:
    p_infectious = pop.F.size / pop.size

    # Similarly, the probability of being a Susceptible agent is:
    p_susceptible = pop.S.size / pop.size

    # The probability that one meeting between two agents has one
    # infectious and one Susceptible is:
    p_one_potential_transmission = p_infectious * p_susceptible

    # Potential transmissions become actual transmissions with
    # probability beta:
    p_one_transmission = p_one_potential_transmission * beta

    # return both probability of transmission and its complement,
    # as Mods will need both

    return [p_one_transmission, 1 - p_one_transmission]


#______________________________________________________________________________
# Define cohorts

pop.make_cohort('S',WN(pop.health.eq,0)) # Susceptible agents
pop.make_cohort('E',WN(pop.health.eq,1)) # Exposed (aymptomatic) agents
pop.make_cohort('I',WN(pop.health.eq,2)) # Infected (symptomatic) agents
pop.make_cohort('R',WN(pop.health.eq,3)) # Recovered agents
pop.make_cohort('D',WN(pop.health.eq,4)) # Deceased agents


pop.make_cohort('F', # InFectious agents
                (lambda : np.logical_and(
                    pop.quarantined.eq(0),
                    (np.logical_or (pop.health.eq(1), pop.health.eq(2))))))

pop.make_cohort('Q', WN(pop.quarantined.eq,1))  # Quarantined agents

# We also will need a "population cohort" whose members are all the agents in the population:
pop.make_cohort('P', (lambda: np.ones(pop.size).astype(bool)), dynamic=False)

#______________________________________________________________________________
# define Mods

# set or reset the health and quarantined attributes to 0
sim.make_mod(
    name= 'initialize_attributes',
    cohort = pop.P,
    mods = [ [WN(pop.health.assign,0), WN(pop.quarantined.assign,0) ] ],
    sim_phase = 'setup'
)

# seed the population with a small fraction of Exposed agents
sim.make_mod(
    name = 'seed_infections',
    cohort = pop.P,
    mods = [[WN(pop.health.assign,1)],
            [WN(pop.health.assign,0)]
            ],
    prob_spec = (.001,.999),
    sim_phase= 'setup'
    )

# This drives the epidemic dynamics:  The probability
# of becoming Exposed depends on p_transmission and beta
sim.make_mod(
    name= 'susceptible_to_exposed',
    cohort = pop.S,
    mods = [[WN(pop.health.assign,1)],
            [] ],
    prob_spec = WN(p_transmission,pop.beta),
    sim_phase = 'loop'
)

# Exposed agents can become Infected, Recovered or remain Exposed
sim.make_mod(
    name = 'exposed_to_next',
    cohort = pop.E,
    mods = [ [WN(pop.health.assign,2)],   # Exposed transition to Infected
             [WN(pop.health.assign,3)],   # Exposed transition to Recovered
             [] ],                          # Do nothing (remain Exposed)
    prob_spec = (.2,.2,.6),
    sim_phase = 'loop'
    )

# Infected agents can become Recovered, Dead or remain Infected
sim.make_mod(
    name = 'infected_to_next',
    cohort = pop.I,
    mods = [ [WN(pop.health.assign,3)],   # Infected transition to Recovered
             [WN(pop.health.assign,4)],   # Infected transition to Dead
             [] ],                          # Do nothing
    prob_spec = (.1,.01,.89),
    sim_phase = 'loop'
    )


# p_quarantine is an aPRAM parameter that contains the probabilities
# of quarantining and doing nothing.   We'll experiment with these
# probabilities to see their effect on epidemic dynamics

pop.p_quarantine.val = [.1,.9]

# This Mod applies to the Infected Cohort only.  In this model, agents
# quarantine only when they show symptoms.  We could also make the
# cohort F, which includes Exposed and Infected individuals, or to
# simulate the benefits of contact tracing we could write other Mods to
# represent knowing that you have been in contact with an infectious agent
# and self-quarantine if you have.

sim.make_mod(
    name = 'quarantine',
    cohort = pop.I,
    mods = [ [WN(pop.quarantined.assign, 1)],
              [] ],
    prob_spec = WN(pop.p_quarantine),
    sim_phase = 'loop'
    )




#______________________________________________________________________________
# set up the data gathering -- this will be easier and
# more intuitive in future versions of aPRAM; ignore it for now

def probe_fn (day):
    record = [day, sim.susceptible_to_exposed.mod_selector.probs[0]]
    record.extend([c.size for c in [pop.S,pop.E,pop.I,pop.R,pop.D,pop.F,pop.Q]])
    return record

probe_labels = ['day','p','S','E','I','R','D','F','Q']



#______________________________________________________________________________
# set up the simulation.  aPRAM has set up a Sim object
# called sim "behind the scenes"; we'll look at it later.
# For now, we need only tell sim how many iterations to run
# and which functions to use to collect data

sim.num_iterations  = 100
sim.probe_labels    = probe_labels
sim.probe_fn        = probe_fn

# Now we will test the efficacy of quarantine.  We'll run
# two simulations, the first with a low probability of
# quarantine, the second with a higher probability.  We'll
# store the data generated by each simulation in a pandas
# frame and then plot it.

# set beta however you want
pop.beta.val = .3
# set a low probability of quarantine, record it for plotting
pop.p_quarantine.val = [.1,.9]
plot_label_0 = str(pop.p_quarantine.val[0])

# tell sim to run
sim.run_simulation()

# store sim.records -- the data -- in a pandas dataframe
df1 = pd.DataFrame(sim.records,columns=probe_labels)

# reset sim.records so we can re-run sim
sim.records=[]

# set a higher probability of quarantine, rerun sim and collect data
pop.p_quarantine.val = [.3,.7]
plot_label_1 = str(pop.p_quarantine.val[0])
sim.run_simulation()
df3 = pd.DataFrame(sim.records,columns=probe_labels)
#sim.records=[]

print("Ready to plot the results...")


# plot the results
ax = df1['F'].plot(label = 'Num. infected, P(quarantine) = '+plot_label_0)
df1['D'].plot(ax=ax, label = 'Num. dead, P(quarantine) = '+plot_label_0)
df3['F'].plot(ax=ax, label = 'Num. infected, P(quarantine) = '+plot_label_1)
df3['D'].plot(ax=ax, label = 'Num. dead, P(quarantine) = '+plot_label_1)
ax.legend()
plt.title(label = 'Epidemic dynamics over '+str(sim.num_iterations)+' days. Beta ='+str(pop.beta.val))
plt.show()

