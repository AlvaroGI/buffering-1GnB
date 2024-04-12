import copy
import itertools
import math
import random
import signal
import time
from time import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats
from scipy.optimize import fsolve
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdmn


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#-------------------------------- PURIFICATION POLICIES -----------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def policy_DEJMPS(F, rho_new, num_new_links):
	'''Purification policy:
		2-to-1: DEJMPS purification protocol.
		x-to-1: uses one new link for DEJMPS and discards the rest.

	Parameters:
	- F:		(int) Fidelity of the buffered state (Werner).
	- rho_new:	(np.array) Density matrix of newly generated entangled links,
							written in the Bell-state basis: 00+11, 00-11, 01+10, 01-10.
							The fidelity is the first entry of the matrix.
	- num_new_links:	(int) Number of newly generated links. The protocol performs
								(num_new_links)-to-1 purification.

	Returns:
	- p_purif_succ:	(float) Probability of success.
	- F_out:	(float) Output fidelity.'''

	assert num_new_links >= 1

	## Werner state in memory ##
	A_werner = F
	B_werner = (1-F)/3
	C_werner = (1-F)/3
	D_werner = (1-F)/3

	## Diagonal elements of the newly generated state (in Bell-state basis) ##
	A = rho_new[0][0]
	B = rho_new[3][3]
	C = rho_new[2][2]
	D = rho_new[1][1]

	#p_purif_succ = (A+B)*(A_werner+B_werner) + (C+D)*(C_werner+D_werner)
	#F_out = (A*A_werner + B*B_werner) / p_purif_succ

	## Purification coefficients ##
	c1 = (2/3) * (A+B-C-D) # Prob of success = c1*(F-1/4) + d1
	d1 = (1/2) * (A+B+C+D)
	a1 = (1/3) * (A-3*B+2*C+2*D) # Shifted output fidelity = (a1*(F-1/4) + b1) / (Prob of success)
	b1 = (1/4) * (2*A-B-2*C-2*D)

	return a1,b1,c1,d1

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------- SIMULATION -----------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def single_run_1GnB(n, p_gen, rho_new, q_purif, purif_policy, pur_after_swap, Gamma, p_cons, t_end, randomseed, burn=None):
	"""
	Simulates the 1GnB entanglement buffer.
	Runs a single realization of the process.

	Parameters:
	- n:		(int) Number of bad memories.
	- p_gen:	(float) Probability of successful entanglement generation
						in each bad memory at each time slot.
	- rho_new:	(np.array) Density matrix of newly generated entangled links,
							written in the Bell-state basis: 00+11, 00-11, 01+10, 01-10.
							The fidelity is the first entry of the matrix.
	- q_purif:	(float) Probability of purifying the link in memory when
						new links are generated.
	- purif_policy:	(function) Returns the purification coefficients a_l, b_l, c_l, d_l:
								Prob of success = c_l*(F-1/4) + d_l
								Shifted output fidelity = (a_l*(F-1/4)+b_l)/(Prob of success).
								Inputs: fidelity F of the buffered link, rho_new,
								and number of new links l.
	- pur_after_swap:	(bool) If True, purification can be immediately performed
								after swapping a new link from a B memory to G.
								Otherwise, the other new links are discarded.
	- Gamma:	(float) Decoherence rate in number of time slots.
	- p_cons:	(float) Probability of consumption request at each time slot.
	- t_end:	(int) Duration of the run in number of time slots is t_end+1.
	- burn:		(float) Burn the first percentage of samples.

	Returns:
	- Fcons_avg:		(float) Average fidelity upon consumption.
	- Fcons_stderr:		(float) Standard error on the average fidelity.
	- A_avg:		(float) Ratio of accepted consumption requests.
	- A_stderr:		(float) Standard error on the availability.
	- buffered_fidelity_trace:	(list of floats) Element i is the fidelity of the buffered
													link at the beginning of time slot i.
	- cons_requests_trace:	(list of bools) Element i is True if there was a consumption
											request at the beginning of time slot i.
											In that case, the consumed fidelity is
											buffered_fidelity_trace[i].
	- purif_events:		(list of ints) Time slots in which purification was attempted.
	"""

	#------------------------------------------------------
	# Check valid parameter values
	#------------------------------------------------------
	assert p_gen>=0 and p_gen<=1
	assert q_purif>=0 and q_purif<=1
	assert p_cons>=0 and p_cons<=1

	if pur_after_swap:
		raise ValueError('pur_after_swap not implemented')
	if burn:
		raise ValueError('burn not implemented')
	
	#------------------------------------------------------
	# Initialize variables
	#------------------------------------------------------
	if randomseed:
		np.random.seed(randomseed)
		random.seed(randomseed)

	# Fidelity of state in memory (None if there is no state)
	F = None

	# Logs
	buffered_fidelity_trace = [None for i in range(t_end+1)]
	cons_requests_trace = (np.random.rand(t_end+1) < p_cons)
	purif_events = []
	cons_fidelities = []
	cons_events = 0

	#------------------------------------------------------
	# Run process
	#------------------------------------------------------
	for t in range(t_end+1):
		buffered_fidelity_trace[t] = F

		# Decohere
		if F is not None:
			F = 0.25 + (F-0.25)*np.exp(-Gamma)

		# Consume and go to next time slot
		if cons_requests_trace[t] and F is not None:
			cons_fidelities += [F]
			cons_events += 1
			F = None
			continue

		# Generate (if consumption was not possible)
		num_new_links = np.random.binomial(n, p_gen)

		if num_new_links > 0:
			# Swap to memory (if memory is empty)
			if F is None:
				F = rho_new[0][0]
				num_new_links -= 1
				if (num_new_links==0) or (not pur_after_swap):
					continue

			# Purify
			if np.random.rand() < q_purif:
				a_l,b_l,c_l,d_l = purif_policy(F,rho_new,num_new_links)
				p_purif_succ = c_l*(F-1/4) + d_l
				F_out = ( (a_l*(F-1/4)+b_l) / p_purif_succ ) + 1/4
				if np.random.rand() < p_purif_succ:
					# Success
					F = F_out
				else:
					# Failure
					F = None
				purif_events += [t]

	# Average consumed fidelity
	Fcons_avg = np.mean(cons_fidelities)
	Fcons_stderr = np.std(cons_fidelities)/np.sqrt(len(cons_fidelities))

	# Availability 
	# 	(standard error for a Bernoulli process: https://en.wikipedia.org/wiki/
	# 	Binomial_proportion_confidence_interval#Standard_error_of_a_proportion_estimation_when_using_weighted_data)
	A_avg = cons_events/sum(cons_requests_trace)
	A_stderr = np.sqrt(A_avg*(1-A_avg)/sum(cons_requests_trace))

	return Fcons_avg, Fcons_stderr, A_avg, A_stderr, buffered_fidelity_trace, cons_requests_trace, purif_events



#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------- PLOTS ----------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
def plot_run_1GnB(Fcons_avg, buffered_fidelity_trace, cons_requests_trace, purif_events, n, p_gen, rho_new, q_purif, purif_policy, pur_after_swap, Gamma, p_cons, t_end, randomseed=None):
	"""
	Plot the fidelity of the buffered memory in the 1GnB entanglement buffer over time,
	for multiple realizations of the process.

	Parameters:
	- Fcons_avg:		(float) Average fidelity upon consumption.
	- buffered_fidelity_trace:	(list of floats) Element i is the fidelity of the buffered
												link at the beginning of time slot i.
	- cons_requests_trace:	(list of bools) Element i is True if there was a consumption
											request at the beginning of time slot i.
											In that case, the consumed fidelity is
											buffered_fidelity_trace[i].
	- purif_events:		(list of ints) Time slots in which purification was attempted.
	- n:		(int) Number of bad memories.
	- p_gen:	(float) Probability of successful entanglement generation
						in each bad memory at each time slot.
	- rho_new:	(np.array) Density matrix of newly generated entangled links,
							written in the Bell-state basis: 00+11, 00-11, 01+10, 01-10.
							The fidelity is the first entry of the matrix.
	- q_purif:	(float) Probability of purifying the link in memory when
						new links are generated.
	- purif_policy:	(function) Returns the purification coefficients a_l, b_l, c_l, d_l:
								Prob of success = c_l*(F-1/4) + d_l
								Shifted output fidelity = (a_l*(F-1/4)+b_l)/(Prob of success).
								Inputs: fidelity F of the buffered link, rho_new,
								and number of new links l.
	- pur_after_swap:	(bool) If True, purification can be immediately performed
								after swapping a new link from a B memory to G.
								Otherwise, the other new links are discarded.
	- Gamma:	(float) Decoherence rate in number of time slots.
	- p_cons:	(float) Probability of consumption request at each time slot.
	- t_end:	(int) Duration of the run in number of time slots is t_end+1.

	Returns:
	- 
	"""
	
	fig, ax = plt.subplots()
	
	# Plot evolution of fidelity
	plt.scatter(range(t_end+1), buffered_fidelity_trace)
	plt.plot(range(t_end+1), buffered_fidelity_trace, color='tab:blue', marker='o', zorder=0)

	# Plot average
	plt.plot([0,t_end+1], [Fcons_avg, Fcons_avg], '--k')

	# Highlight consumption events
	cons_events = []
	F_cons_events = []
	for t, consumption in enumerate(cons_requests_trace):
		if consumption:
			cons_events += [t]
			F_cons_events += [buffered_fidelity_trace[t]]
	plt.scatter(cons_events, F_cons_events, color='k', marker='x', zorder=1, label='Consumption events')
	
	# Highlight purification events
	purif_events = purif_events
	plt.scatter(purif_events, [buffered_fidelity_trace[t] for t in purif_events],
				color='tab:orange', marker='^', zorder=1, label='Purification events')
		


	#------------------------------------------------------
	# Plot specs
	#------------------------------------------------------

	# Labels
	plt.xlabel('Time')
	plt.ylabel('Fidelity')

	# Axes limits
	plt.xlim(0,t_end)
	plt.ylim(0.25,1)

	# Ticks
	ax.set_yticks([0.25, 0.5, 0.75, 1])

	# Legend
	plt.legend()













