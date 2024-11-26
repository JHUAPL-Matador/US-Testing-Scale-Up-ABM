import datetime, math, os, pathlib, sys
from collections import Counter
from functools import partial

import joblib
import pandas as pd
import numpy as np
import pylab as pl

import sciris as sc
import covasim as cv
import covasim.parameters as cvp

from covasim import misc as cvm
from covasim import utils as cvu
from covasim import base as cvb
from covasim import defaults as cvd
from covasim import immunity as cvi
from covasim.interventions import preprocess_day, process_daily_data, get_day, get_quar_inds

from numpy.random import default_rng



task_id = sys.argv[1]

print("imports succeeded...")


# see https://docs.idmod.org/projects/covasim/en/latest/tutorials/tut_analyzers.html
class store_doses(cv.Analyzer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # This is necessary to initialize the class properly
        self.t = []
        self.n_complete_primary = []
        return

    def apply(self, sim):
        ppl = sim.people # Shorthand
        self.t.append(sim.t)
        # see boosters tutorial
        # https://docs.idmod.org/projects/covasim/en/latest/tutorials/tut_immunity.html
        self.n_complete_primary.append((ppl.doses >= 2).sum())
        return

    def plot(self):
        plt.figure()
        plt.plot(self.t, self.n_complete_primary, label='completed primary series')
        plt.legend()
        plt.ylabel('people')
        sc.setylim() # Reset y-axis to start at 0
        sc.commaticks() # Use commas in the y-axis labels
        return
    
    
    
def load_daily_vax_doses(mappers_path):
    # load dicts mapping date to number vax doses
    [dt2pfizer_primary, dt2moderna_primary, dt2janssen_primary, dt2boost] = joblib.load(mappers_path)
    
    # sanity checks: for this to work, need our dates (keys) to be strings
    assert type(list(dt2pfizer_primary.keys())[0]) == type('foo')
    assert type(list(dt2boost.keys())[0]) == type('foo')
    
    # also make sure that dicts do not return NaNs
    for ix, d in enumerate([
        dt2pfizer_primary, dt2moderna_primary, dt2janssen_primary, dt2boost
    ]):
        for k, v in d.items():
            if math.isnan(v):
                print(k,v)
                break
                
    return dt2pfizer_primary, dt2moderna_primary, dt2janssen_primary, dt2boost


# CUSTOM INTERVENTIONS / METHODS / FNCS vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

def define_subgroups(
    pop_size=50e3, 
    FRAC_NONTESTERS=0.0,
    FRAC_NONISOLATORS=0.0,
    FRAC_SXISOLATORS=0.0,
    seed=314
):
    """Given pop_size (number of agents) and fractions, partition agents into 4 subsets:
    non-testers (agents who will never accept test), non-isolators (agents that will
    never isolate even with a positive test), symptomatic-isolators (agents that
    isolate if they have symptoms, even before tested)"""
    
    assert 0 <= FRAC_NONTESTERS <= 1.0
    assert 0 <= FRAC_NONISOLATORS <= 1.0
    assert 0 <= FRAC_SXISOLATORS <= 1.0
    assert FRAC_NONTESTERS + FRAC_NONISOLATORS + FRAC_SXISOLATORS <= 1.0, \
        f"sum > 1: {FRAC_NONTESTERS=} {FRAC_NONISOLATORS=} {FRAC_SXISOLATORS=}"
    
    idx_nontesters = int(FRAC_NONTESTERS * pop_size)
    idx_nonisolators = idx_nontesters + int(FRAC_NONISOLATORS * pop_size)
    idx_sxisolators = idx_nonisolators + int(FRAC_SXISOLATORS * pop_size)

    # this represents the IDs of agents in covasim
    uid_copy = np.arange(pop_size, dtype=np.int32)
    
    # shuffle uid list, split at indices corresponding to input fractions
    # 'default' is list of indices not in nontesters / nonisolators / sxisolators
    rng = default_rng(seed=seed)
    rng.shuffle(uid_copy)
    nontesters, nonisolators, sxisolators, default = \
        np.split(uid_copy, [idx_nontesters, idx_nonisolators, idx_sxisolators])

    return nontesters, nonisolators, sxisolators, default



def dynamic_agent_partition(
    sim,
    nontester_fracs=[0.0, 0.0], 
    nonisolator_fracs=[0.0, 0.0], 
    sxisolator_fracs=[0.0, 0.0],
    days=[0,45]
):
    """Based on first example under Custom interventions at
    https://docs.idmod.org/projects/covasim/en/latest/tutorials/tut_interventions.html"""
    pop_size = sim.pars['pop_size']
    for i in range(len(days)):
        # if we are on one of the days specified as input, repartition agents &
        # update respective attributes on people instance
        # note: if seed IS NOT changed, the shuffle step to create partitions is same;
        # so, if fractions go down over time, subset of agents will change partitions as follows:
        # nontesters -> nonisolators, nonisolators -> sxisolators, sxisolators -> defaults;
        # if fractions go up over time, movement is in opposite direction;
        # if seed IS changed, the shuffle step to create partitions is different,
        # and thus new partition is created with no relation to prior
        if sim.t == days[i]:
            nontesters, nonisolators, sxisolators, default = define_subgroups(
                pop_size, 
                # setting seed=i provides new seed for each partition, 
                # but these new partitions will be constant across replicate runs;
                # alternatively could set seed as constant, e.g., 314,
                # this would ensure shuffle step is always same, only fractions change
                seed=i,
                FRAC_NONTESTERS=nontester_fracs[i], 
                FRAC_NONISOLATORS=nonisolator_fracs[i],  
                FRAC_SXISOLATORS=sxisolator_fracs[i], 
            )
            sim.people.nontesters = nontesters
            sim.people.sxisolators = sxisolators
            sim.people.nonisolators = nonisolators

    return



class test_num_modified(cv.Intervention):
    """
    Create custom intervention that is slight tweak of original covasim test_num intervention
    (see https://docs.idmod.org/projects/covasim/en/latest/_modules/covasim/interventions.html#test_num).
    Only change is: we reset the nontesters prob of testing to zero, just prior to
    deciding which agents get tested.
    
    Lines added shown with '##' comment just above. 
    TODO: consider subclassing covasim.interventions.test_num & overriding 'apply' method
    """

    def __init__(self, daily_tests, symp_test=100.0, quar_test=1.0, quar_policy=None, subtarget=None,
                 ili_prev=None, sensitivity=1.0, loss_prob=0, test_delay=0,
                 start_day=0, end_day=None, swab_delay=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.label       = 'test_num_modified'
        self.daily_tests = daily_tests # Should be a list of length matching time
        self.symp_test   = symp_test   # Set probability of testing symptomatics
        self.quar_test   = quar_test # Probability of testing people in quarantine
        self.quar_policy = quar_policy if quar_policy else 'start'
        self.subtarget   = subtarget  # Set any other testing criteria
        self.ili_prev    = ili_prev     # Should be a list of length matching time or a float or a dataframe
        self.sensitivity = sensitivity
        self.loss_prob   = loss_prob
        self.test_delay  = test_delay
        self.start_day   = start_day
        self.end_day     = end_day
        self.pdf         = cvu.get_pdf(**sc.mergedicts(swab_delay)) # If provided, get the distribution's pdf -- this returns an empty dict if None is supplied
        return


    def initialize(self, sim):
        ''' Fix the dates and number of tests '''

        # Handle days
        super().initialize()

        self.start_day   = preprocess_day(self.start_day, sim)
        self.end_day     = preprocess_day(self.end_day,   sim)
        self.days        = [self.start_day, self.end_day]

        # Process daily data
        self.daily_tests = process_daily_data(self.daily_tests, sim, self.start_day)
        self.ili_prev    = process_daily_data(self.ili_prev,    sim, self.start_day)


    def finalize(self, sim):
        ''' Ensure variables with large memory footprints get erased '''
        super().finalize()
        self.subtarget = None # Reset to save memory
        return


    def apply(self, sim):

        t = sim.t
        start_day = get_day(self.start_day, self, sim)
        end_day   = get_day(self.end_day,   self, sim)
        if t < start_day:
            return
        elif end_day is not None and t > end_day:
            return

        # Check that there are still tests
        rel_t = t - start_day
        if rel_t < len(self.daily_tests):
            n_tests = sc.randround(self.daily_tests[rel_t]/sim.rescale_vec[t]) # Correct for scaling that may be applied by rounding to the nearest number of tests
            if not (n_tests and pl.isfinite(n_tests)): # If there are no tests today, abort early
                return
            else:
                sim.results['new_tests'][t] += n_tests
        else:
            return

        test_probs = np.ones(sim.n) # Begin by assigning equal testing weight (converted to a probability) to everyone
        
        # Calculate test probabilities for people with symptoms
        symp_inds = cvu.true(sim.people.symptomatic)
        symp_test = self.symp_test
        if self.pdf: # Handle the onset to swab delay
            symp_time = cvd.default_int(t - sim.people.date_symptomatic[symp_inds]) # Find time since symptom onset
            inv_count = (np.bincount(symp_time)/len(symp_time)) # Find how many people have had symptoms of a set time and invert
            count = np.nan * np.ones(inv_count.shape) # Initialize the count
            count[inv_count != 0] = 1/inv_count[inv_count != 0] # Update the counts where defined
            symp_test *= self.pdf.pdf(symp_time) * count[symp_time] # Put it all together

        test_probs[symp_inds] *= symp_test # Update the test probabilities

        # Handle symptomatic testing, taking into account prevalence of ILI symptoms
        if self.ili_prev is not None:
            if rel_t < len(self.ili_prev):
                n_ili = int(self.ili_prev[rel_t] * sim['pop_size'])  # Number with ILI symptoms on this day
                ili_inds = cvu.choose(sim['pop_size'], n_ili) # Give some people some symptoms. Assuming that this is independent of COVID symptomaticity...
                ili_inds = np.setdiff1d(ili_inds, symp_inds)
                test_probs[ili_inds] *= self.symp_test

        # Handle quarantine testing
        quar_test_inds = get_quar_inds(self.quar_policy, sim)
        test_probs[quar_test_inds] *= self.quar_test

        # Handle any other user-specified testing criteria
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
            test_probs[subtarget_inds] = test_probs[subtarget_inds]*subtarget_vals

        # Don't re-diagnose people
        diag_inds  = cvu.true(sim.people.diagnosed)
        test_probs[diag_inds] = 0.0

        # With dynamic rescaling, we have to correct for uninfected people outside of the population who would test
        if sim.rescale_vec[t]/sim['pop_scale'] < 1: # We still have rescaling to do
            in_pop_tot_prob = test_probs.sum()*sim.rescale_vec[t] # Total "testing weight" of people in the subsampled population
            out_pop_tot_prob = sim.scaled_pop_size - sim.rescale_vec[t]*sim['pop_size'] # Find out how many people are missing and assign them each weight 1
            in_frac = in_pop_tot_prob/(in_pop_tot_prob + out_pop_tot_prob) # Fraction of tests which should fall in the sample population
            n_tests = sc.randround(n_tests*in_frac) # Recompute the number of tests
            
        ## TEST PROBS FOR NONTESTERS GET REASSIGNED TO 0
        test_probs[sim.people.nontesters] = 0.0

        # Now choose who gets tested and test them
        n_tests = min(n_tests, (test_probs!=0).sum()) # Don't try to test more people than have nonzero testing probability
        test_inds = cvu.choose_w(probs=test_probs, n=n_tests, unique=True) # Choose who actually tests (returns list of agents (identified by index))
        sim.people.test(test_inds, test_sensitivity=self.sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay)

        ## NO PERSON CHOSEN FOR TESTING SHOULD BE AMONG NONTESTERS
        assert set(test_inds).intersection(sim.people.nontesters)==set()
        
        return test_inds

    

class contact_tracing_modified(cv.Intervention):
    """
    Create custom intervention that is slight tweak of original covasim contact_tracing intervention
    (see https://docs.idmod.org/projects/covasim/en/latest/_modules/covasim/interventions.html#contact_tracing).
    Only change is: we do not notify any nontesters / nonisolators of any sick contacts. We want to
    ensure these agents do NOT change behavior in response to contact tracing -- simplest way to enforce
    seems to be 'just do not notify them'.
    
    As with original contact_tracing intervention, without testing, contact tracing has no effect.
    
    Lines added shown with '##' comment just above. 
    TODO: consider subclassing covasim.interventions.contact_tracing & overriding 'notify_contacts' method
    """
    
    def __init__(self, trace_probs=None, trace_time=None, start_day=0, end_day=None, presumptive=False, quar_period=None, capacity=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.trace_probs = trace_probs
        self.trace_time  = trace_time
        self.start_day   = start_day
        self.end_day     = end_day
        self.presumptive = presumptive
        self.capacity = capacity
        self.quar_period = quar_period # If quar_period is None, it will be drawn from sim.pars at initialization
        return


    def initialize(self, sim):
        ''' Process the dates and dictionaries '''
        super().initialize()
        self.start_day = preprocess_day(self.start_day, sim)
        self.end_day   = preprocess_day(self.end_day,   sim)
        self.days      = [self.start_day, self.end_day]
        if self.trace_probs is None:
            self.trace_probs = 1.0
        if self.trace_time is None:
            self.trace_time = 0.0
        if self.quar_period is None:
            self.quar_period = sim['quar_period']
        if sc.isnumber(self.trace_probs):
            val = self.trace_probs
            self.trace_probs = {k:val for k in sim.people.layer_keys()}
        if sc.isnumber(self.trace_time):
            val = self.trace_time
            self.trace_time = {k:val for k in sim.people.layer_keys()}
        return


    def apply(self, sim):
        '''
        Trace and notify contacts

        Tracing involves three steps that can independently be overloaded or extended
        by derived classes

        - Select which confirmed cases get interviewed by contact tracers
        - Identify the contacts of the confirmed case
        - Notify those contacts that they have been exposed and need to take some action
        '''
        t = sim.t
        start_day = get_day(self.start_day, self, sim)
        end_day   = get_day(self.end_day,   self, sim)
        if t < start_day:
            return
        elif end_day is not None and t > end_day:
            return

        trace_inds = self.select_cases(sim)
        contacts = self.identify_contacts(sim, trace_inds)
        self.notify_contacts(sim, contacts)
        return contacts


    def select_cases(self, sim):
        '''
        Return people to be traced at this time step
        '''
        if not self.presumptive:
            inds = cvu.true(sim.people.date_diagnosed == sim.t) # Diagnosed this time step, time to trace
        else:
            just_tested = cvu.true(sim.people.date_tested == sim.t) # Tested this time step, time to trace
            inds = cvu.itruei(sim.people.exposed, just_tested) # This is necessary to avoid infinite chains of asymptomatic testing

        # If there is a tracing capacity constraint, limit the number of agents that can be traced
        if self.capacity is not None:
            capacity = int(self.capacity / sim.rescale_vec[sim.t])  # Convert capacity into a number of agents
            if len(inds) > capacity:
                inds = np.random.choice(inds, capacity, replace=False)

        return inds


    def identify_contacts(self, sim, trace_inds):
        '''
        Return contacts to notify by trace time

        In the base class, the trace time is the same per-layer, but derived classes might
        provide different functionality e.g. sampling the trace time from a distribution. The
        return value of this method is a dict keyed by trace time so that the `Person` object
        can be easily updated in `contact_tracing.notify_contacts`

        Args:
            sim: Simulation object
            trace_inds: Indices of people to trace

        Returns: {trace_time: np.array(inds)} dictionary storing which people to notify
        '''

        if not len(trace_inds):
            return {}

        contacts = sc.ddict(list)

        for lkey, this_trace_prob in self.trace_probs.items():

            if this_trace_prob == 0:
                continue

            traceable_inds = sim.people.contacts[lkey].find_contacts(trace_inds)
            if len(traceable_inds):
                contacts[self.trace_time[lkey]].extend(cvu.binomial_filter(this_trace_prob, traceable_inds)) # Filter the indices according to the probability of being able to trace this layer

        array_contacts = {}
        for trace_time, inds in contacts.items():
            array_contacts[trace_time] = np.fromiter(inds, dtype=cvd.default_int)

        return array_contacts


    def notify_contacts(self, sim, contacts):
        '''
        Notify contacts

        This method represents notifying people that they have had contact with a confirmed case.
        In this base class, that involves

        - Setting the 'known_contact' flag and recording the 'date_known_contact'
        - Scheduling quarantine

        Args:
            sim: Simulation object
            contacts: {trace_time: np.array(inds)} dictionary storing which people to notify
        '''
        is_dead = cvu.true(sim.people.dead) # Find people who are not alive
        for trace_time, contact_inds in contacts.items():
            contact_inds = np.setdiff1d(contact_inds, is_dead) # Do not notify contacts who are dead
            
            ## DO NOT NOTIFY CONTACTS WHO ARE NONISOLATORS / NONTESTERS
            ## (THESE GUYS WILL NOT CHANGE BEHAVIOR IF CONTACT TRACED; SO JUST ENFORCE THIS BY
            ## NOT NOTIFYING THEM) -- comment these lines out to experiment with this intervention
            contact_inds = np.setdiff1d(contact_inds, sim.people.nonisolators).astype(np.int32) # ensure ints
            contact_inds = np.setdiff1d(contact_inds, sim.people.nontesters).astype(np.int32) # ensure ints
            
            sim.people.known_contact[contact_inds] = True
            sim.people.date_known_contact[contact_inds] = np.fmin(sim.people.date_known_contact[contact_inds], sim.t + trace_time)
            sim.people.schedule_quarantine(contact_inds, start_date=sim.t + trace_time, period=self.quar_period - trace_time)  # Schedule quarantine for the notified people to start on the date they will be notified
        return
    
    
    
def check_enter_iso_modified(self):
    """Custom fnc to account for sxisolators and nonisolators
    (see https://docs.idmod.org/projects/covasim/en/latest/_modules/covasim/people.html#People.check_enter_iso)"""
    
    # here we modify final list of isolated individuals --
    # 1. determine initial list of individuals to isolate (iso inds) in standard way
    iso_inds = cvu.true(self.date_diagnosed == self.t)
    
    # below is from original function
    #self.isolated[iso_inds] = True
    #self.date_end_isolation[iso_inds] = self.date_recovered[iso_inds]
    #return iso_inds

    # 2. find agents that became symptomatic on given date & add sxisolators subset to iso inds
    current_sx = cvu.true((self.date_symptomatic == self.t))
    current_sx_isolators = np.intersect1d(current_sx, self.sxisolators) 
    iso_inds_updated = np.union1d(iso_inds, current_sx_isolators).astype(np.int32) # ensure ints
    
    # 3. remove nonisolators from initial list of iso inds
    iso_inds_updated = np.setdiff1d(iso_inds_updated, self.nonisolators).astype(np.int32) # ensure ints
    
    # TODO: think of a better way to test & move test outside this fnc
    assert set(self.nonisolators).intersection(iso_inds_updated)==set()
    assert set(current_sx_isolators).intersection(iso_inds_updated)==set(current_sx_isolators)

    # 4. from here down, we use new list of iso inds and code from original function
    self.isolated[iso_inds_updated] = True
    self.date_end_isolation[iso_inds_updated] = self.date_recovered[iso_inds_updated]
    return iso_inds_updated


# CUSTOM INTERVENTIONS / METHODS / FNCS ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def make_sim(
    seed, beta, data_path, mappers_path,
    N_AGENTS = 100e3,
    people_per_agent = 1, 
    start_day = '2020-01-20',
    end_day='2022-03-01', 
    verbose=0,
    analyzers=[],
    pop_infected=10,
    dynamic_agent_partition=dynamic_agent_partition,
):

    # set parameter vals / rename using covasim varnames
    #total_pop    = 330e6 # ~ US population size
    pop_type     = 'hybrid'
    pop_size     = N_AGENTS # "pop_size" actually refers to number of agents in sim
    pop_scale    = people_per_agent # int(total_pop/pop_size)
    asymp_factor = 1                                       # Multiply beta by this factor for asymptomatic cases
    # contacts     = {'h':2.0, 's':20, 'w':16, 'c':20}     # defaults for contacts / layer are overwritten with data for 'usa'
    
    # check on sim being initialized
    print(f"simulating {N_AGENTS=}, {people_per_agent=}, total popn {N_AGENTS * people_per_agent}")
    print(f"using data from {data_path}")
    print(f"using daily vax doses from {mappers_path}")
    
    # load vax per day data
    dt2pfizer_primary, dt2moderna_primary, dt2janssen_primary, dt2boost = load_daily_vax_doses(mappers_path)
      
    # make dict for instantiating sim object
    pars = sc.objdict(
        use_waning   = True,
        pop_size     = pop_size,     # =N_AGENTS, input parameter
        pop_infected = pop_infected, # input parameter, 1000 usually
        pop_scale    = pop_scale,    # =people_per_agent, input parameter
        pop_type     = pop_type,     # 'hybrid'
        start_day    = start_day,    # input parameter, 2020-01-20 usually
        end_day      = end_day,      # input parameter, 2022-03-01 usually
        beta         = beta,         # input parameter ~ 0.0079-0.01
        asymp_factor = asymp_factor, # set above, 
        # contacts     = contacts,   # instead, use defaults for 'usa': {'h': 1.491, 's': 20, 'w': 16, 'c': 20} 
        rescale      = True,
        rand_seed    = seed,
        verbose      = verbose,
    )

    # instantiate sim object
    sim = cv.Sim(pars=pars, datafile=data_path, location='usa', analyzers=analyzers)

    # MASKING / DISTANCING::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # this dict has dates as keys, vals are list of relative transmissibility in 4 settings:
    # home, school, work, community; initial "data/theoretically derived" values are in comments
    # to the right, lines with no comment have been added
    t = pd.read_csv('data/param.soc.csv')
    beta_past = sc.odict(dict(zip(t['date'], t.iloc[:,1:-1].values.tolist())))
    print(f"social params:\n{beta_past}")

    beta_dict = beta_past
    beta_days = list(beta_dict.keys())
    h_beta = cv.change_beta(days=beta_days, changes=[c[0] for c in beta_dict.values()], layers='h')
    s_beta = cv.change_beta(days=beta_days, changes=[c[1] for c in beta_dict.values()], layers='s')
    w_beta = cv.change_beta(days=beta_days, changes=[c[2] for c in beta_dict.values()], layers='w')
    c_beta = cv.change_beta(days=beta_days, changes=[c[3] for c in beta_dict.values()], layers='c')
    
    interventions = [h_beta, w_beta, s_beta, c_beta]
    # MASKING / DISTANCING::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    # DYNAMIC AGENT PARTITIONING::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    ## NEW INTERV
    dynamic_agent_partition=partial(
        dynamic_agent_partition, 
        nontester_fracs=[0.0, 0.05, 0.2],
        nonisolator_fracs=[0.0, 0.02, 0.1],
        sxisolator_fracs=[0.0, 0.3, 0.2],
        # change agent partition on these days: ('2020-01-20', '2020-05-01', '2021-05-01'), 
        # can verify by checking sim.date(0), sim.date(102), sim.date(467)
        days=[0, 102, 467],  
    )
    interventions += [dynamic_agent_partition]
    # DYNAMIC AGENT PARTITIONING::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # TESTING/CONTACT TRACING:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # we use actual number of reported tests and defaults for relative transmissibility changes
    # here are isolation / quarantine factors for 'hybrid' popn
    # iso_factor  = dict(h=0.3, s=0.1, w=0.1, c=0.1),  # Multiply beta by this factor for people in isolation
    # quar_factor = dict(h=0.6, s=0.2, w=0.2, c=0.2),  # Multiply beta by this factor for people in quarantine
    # assumed quarantine factor=0.3
    ## NEW INTERV
    testing = test_num_modified(
        daily_tests='data',
        sensitivity=0.85    ## change from default 1.0
    )
    
    ## NEW INTERV
    contact_trace = contact_tracing_modified(
        trace_probs=dict(h=1.0, s=0.3, w=0.2, c=0.1), 
        trace_time=dict(h=0, s=2, w=3, c=5),
        do_plot=False
    )
    
    ## optionally reset iso_factor / quar_factor from defaults for home / school / work / community
    sim.pars['iso_factor'] = dict(h=0.3, s=0.1, w=0.1, c=0.1)
    sim.pars['quar_factor'] = dict(h=0.6, s=0.2, w=0.2, c=0.2)
    
    interventions += [testing, contact_trace]
    # TESTING/CONTACT TRACING:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    # VACCINES::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # we use actual number of reported vaccinations, defaults are in 
    # covasim.parameters.get_vaccine_variant_pars()
    # first make vaccine specific dicts with params, this essentially just adds info for omicron for
    # the 3 different vaccines
    dose_pars = cvp.get_vaccine_dose_pars()['pfizer']
    variant_pars = cvp.get_vaccine_variant_pars()['pfizer']
    variant_pars['omicron'] = 0.1 # 0.8, some but not much protection from primary series
    variant_pars['label'] = 'pfizer-vax'
    pfizer_vaccine = sc.mergedicts({'label':'pfizer_us'}, sc.mergedicts(dose_pars, variant_pars))

    dose_pars = cvp.get_vaccine_dose_pars()['moderna']
    variant_pars = cvp.get_vaccine_variant_pars()['moderna']
    variant_pars['omicron'] = 0.1 # 0.8
    variant_pars['label'] = 'moderna-vax'
    moderna_vaccine = sc.mergedicts({'label':'moderna_us'}, sc.mergedicts(dose_pars, variant_pars))

    dose_pars = cvp.get_vaccine_dose_pars()['jj']
    variant_pars = cvp.get_vaccine_variant_pars()['jj']
    variant_pars['omicron'] = 0.1 # 0.8
    variant_pars['label'] = 'jj-vax'
    jj_vaccine = sc.mergedicts({'label':'jj_us'}, sc.mergedicts(dose_pars, variant_pars)) 
    
    # these functions take date and return number of primary doses vaccine for that date by mfr
    def num_doses_pfizer(sim):
        return int(dt2pfizer_primary.get(sim.date(sim.t), 0))

    def num_doses_moderna(sim):
        return int(dt2moderna_primary.get(sim.date(sim.t), 0))

    def num_doses_janssen(sim):
        return int(dt2janssen_primary.get(sim.date(sim.t), 0))

    # make primary vaccine interventions
    pfizer = cv.vaccinate_num(
        vaccine=pfizer_vaccine, sequence='age', num_doses=num_doses_pfizer,
    )
    moderna = cv.vaccinate_num(
        vaccine=moderna_vaccine, sequence='age', num_doses=num_doses_moderna,
    )
    jj = cv.vaccinate_num(
        vaccine=jj_vaccine, sequence='age', num_doses=num_doses_janssen
    )
    interventions += [pfizer, moderna, jj]
    
    # this function takes date and returns number of booster doses for that date
    def num_boosters(sim):
        return int(dt2boost.get(sim.date(sim.t), 0))
    
    # target boosters to people who have had 2 prior doses of vaccine
    booster_target  = {'inds': lambda sim: cv.true(sim.people.doses != 2), 'vals': 0}
    
    # as an approximation, assume all boosters are pfizer -- update omicron param
    dose_pars = cvp.get_vaccine_dose_pars()['pfizer']
    variant_pars = cvp.get_vaccine_variant_pars()['pfizer']
    variant_pars['omicron'] = 0.34 #0.5
    variant_pars['label'] = 'pfizer-boost'
    pfizer_vaccine_boost = sc.mergedicts({'label':'pfizer_us'}, sc.mergedicts(dose_pars, variant_pars))
    
    # make booster intervention
    pfizer_boost = cv.vaccinate_num(
       vaccine=pfizer_vaccine_boost, sequence='age', booster=True, num_doses=num_boosters,
       subtarget=booster_target, 
    )
    interventions += [pfizer_boost]
    # VACCINES::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    # VARIANTS::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # for simplicity, we model wild type, alpha, delta, omicron thru 3/1/2022 
    # see https://www.medrxiv.org/content/10.1101/2021.12.28.21268491v2.full#F2, Figure 2E for potential
    # parameters on immunity from prior infection, vaccine
    var_param_names = [
        'rel_beta', 'rel_symp_prob', 'rel_severe_prob', 'rel_crit_prob', 'rel_death_prob'
    ]
    v = pd.read_csv('data/param.var.csv')
    print(f"variant params:\n{v.T}")
    assert v.iloc[0,0]=='wildtype'
    assert v.iloc[1,0]=='alpha'
    assert v.iloc[2,0]=='delta'
    assert v.iloc[3,0]=='omicron'
    wt = v.iloc[0,1:].values
    al = v.iloc[1,1:].values
    de = v.iloc[2,1:].values
    om = v.iloc[3,1:].values
    
    # adjust hosp, critical, deaths from wildtype here
    for ix, feat in enumerate(var_param_names):
        sim.pars[feat] = wt[ix]
        
    # first wt wave handled through pop_infected parameter (=1000 at beginning of sim)
        
    # second wt wave ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    wt2 = cv.variant(
        label='wild-type-wave2',
        n_imports=3300*100/5,
        days=np.array([
            sim.day('2020-06-01'),sim.day('2020-06-02'),sim.day('2020-06-03'),sim.day('2020-06-04'),sim.day('2020-06-05'),
        ]),
        variant={
            var_param_names[i]: wt[i] for i in range(len(var_param_names))
        }
    )
    wt2.rescale = True
    sim['variants'] += [wt2]
    # second wt wave ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    # third wt wave ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    wt3 = cv.variant(
        label='wild-type-wave3',
        n_imports=3300*100/5,
        days=np.array([
            sim.day('2020-10-15'),sim.day('2020-10-16'),sim.day('2020-10-17'),sim.day('2020-10-18'),sim.day('2020-10-19'),
            sim.day('2020-10-20'),sim.day('2020-10-21'),sim.day('2020-10-22'),sim.day('2020-10-23'),sim.day('2020-10-24'),
        ]),
        variant={
            var_param_names[i]: wt[i] for i in range(len(var_param_names))
        }
    )
    wt3.rescale = True
    sim['variants'] += [wt3]
    # third wt wave ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # Alpha strain from October 2020; n_imports, rel_beta and rel_severe_beta from calibration
    # Alpha: Dec 21 2020 – Jun 21 2021
    alpha = cv.variant(
        'alpha', 
        n_imports=3300*100/5,  
        days=np.array([
            sim.day('2020-12-28'),sim.day('2020-12-29'),sim.day('2020-12-30'),sim.day('2020-12-31'),sim.day('2021-01-01'), 
        ]), 
    )
    alpha.rescale = True
    for ix, feat in enumerate(var_param_names):
        alpha.p[feat] = al[ix]
    sim['variants'] += [alpha]
    
    # Delta strain starting middle of April 2021; n_imports, rel_beta and rel_severe_beta from calibration
    # Delta: Apr 12 2021 – Feb 14 2022
    delta = cv.variant(
        'delta', 
        n_imports=3300*100/10,
        days=np.array([
            sim.day('2021-04-10'),sim.day('2021-04-11'),sim.day('2021-04-12'),sim.day('2021-04-13'),sim.day('2021-04-14'),
            sim.day('2021-04-15'),sim.day('2021-04-16'),sim.day('2021-04-17'),sim.day('2021-04-18'),sim.day('2021-04-19'),
        ]), 
    )
    delta.rescale = True
    for ix, feat in enumerate(var_param_names):
        delta.p[feat] = de[ix]
    sim['variants'] += [delta]
    
    # Add Omicron strain starting middle of Nov 2021; n_imports, rel_beta and rel_severe_beta from calibration
    # Omicron: Dec 6 2021 – Mar 28 2022
    omicron = cv.variant(
        label='omicron', 
        n_imports=3300*100/10,
        days=np.array([
            sim.day('2021-11-26'),sim.day('2021-11-27'),sim.day('2021-11-28'),#sim.day('2021-11-29'),sim.day('2021-11-30'),
            #sim.day('2021-12-01'),sim.day('2021-12-02'),sim.day('2021-12-03'),sim.day('2021-12-04'),sim.day('2021-12-05'),
        ]), 
        variant={
            var_param_names[i]: om[i] for i in range(len(var_param_names))
        }
    )
    omicron.rescale = True
    sim['variants'] += [omicron]
    # VARIANTS::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    

    # add daily age stats analyzer
    analyzers += [cv.daily_age_stats(edges= [0, 30,  65,  80, 100])]

    # update parameters, add analyzers
    sim.update_pars(interventions=interventions, analyzers=analyzers)

    # if desired, change death and critical probabilities, example below
    # interventions += [cv.dynamic_pars({'rel_death_prob':{'days':sim.day('2020-07-01'), 'vals':0.6}})]
    # update the parameters
    # sim.update_pars(interventions=interventions)
    
    # turn off intervention plotting
    for intervention in sim['interventions']:
        intervention.do_plot = False

    sim.initialize()
    
    ## override the check_enter_iso method on people instance with custom method above
    sim.people.check_enter_iso = check_enter_iso_modified.__get__(sim.people, cv.People)
    
    # display the variant params
    var_param_names = [
        'rel_beta', 'rel_symp_prob', 'rel_severe_prob', 'rel_crit_prob', 'rel_death_prob'
    ]

    return sim




# make s0 for parallelization attempt
s0 = make_sim(
    beta=0.016, 
    pop_infected=1000, #1000,
    N_AGENTS=1_000_000, #100_000,
    people_per_agent=330, #3300,
    start_day='2020-01-20',
    end_day='2022-12-31', 
    seed=30, # this gets changed for each replicate
    data_path='data/usa-data-1.0-1.8.csv',
    mappers_path='data/mappers-1.0.joblib',
    verbose=0.1, 
    analyzers=[store_doses(label='doses')]
)

print("made s0...")
s0['rand_seed'] = task_id
s0.set_seed()
s0.label = f"Sim_{task_id}"
print(f"set s0 rand seed to {task_id}...")
s0.run()
print("s0 finished running...")
s0.save(f'results/{s0.label}.C3.pkl')
print("s0 saved in results...done")
#print(s0.pars)
