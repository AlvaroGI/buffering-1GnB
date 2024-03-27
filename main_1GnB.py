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
#------------------------------- PURIFICATION PROTOCOLS -----------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def DEJMPS(F, rho_new, num_new_links):
	"""
	DEJMPS 2-to-1 purification protocol.
	For any num_new_links, uses one new link for DEJMPS and discards the rest.

	Parameters:
	- F:		(int) Fidelity of the buffered state (Werner).
	- rho_new:	(np.array) Density matrix of newly generated entangled links,
							written in the Bell-state basis: 00+11, 00-11, 01+10, 01-10.
							The fidelity is the first entry of the matrix.
	- num_new_links:	(int) Number of newly generated links. The protocol performs
								(num_new_links)-to-1 purification.

	Returns:
	- p_purif_succ:	(float) Probability of success.
	- F_out:	(float) Output fidelity.
	"""
	assert num_new_links >= 1

	A_werner = F
	B_werner = (1-F)/3
	C_werner = (1-F)/3
	D_werner = (1-F)/3

	A = rho_new[0][0]
	B = rho_new[3][3]
	C = rho_new[2][2]
	D = rho_new[1][1]

	p_purif_succ = (A+B)*(A_werner+B_werner) + (C+D)*(C_werner+D_werner)
	F_out = (A*A_werner + B*B_werner) / p_purif_succ

	return p_purif_succ, F_out

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------- SIMULATION -----------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def single_run_1GnB(n, p_gen, rho_new, q_purif, purif_protocol, pur_after_swap, Gamma, p_cons, t_end, randomseed, burn=None):
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
	- purif_protocol:	(function) Returns the probability of success and the
									output fidelity of the purification protocol,
									for a given fidelity F of the buffered link,
									rho_new, and number of new links l.
	- pur_after_swap:	(bool) If True, purification can be immediately performed
								after swapping a new link from a B memory to G.
								Otherwise, the other new links are discarded.
	- Gamma:	(float) Decoherence rate in number of time slots.
	- p_cons:	(float) Probability of consumption request at each time slot.
	- t_end:	(int) Duration of the run in number of time slots is t_end+1.
	- burn:		(float) Burn the first percentage of samples.

	Returns:
	- cons_fid_avg:		(float) Average fidelity upon consumption.
	- cons_fid_stderr:		(float) Standard error on the average fidelity.
	- availability_avg:		(float) Ratio of accepted consumption requests.
	- availability_stderr:	(float) Standard error on the availability.
	- buffered_fidelity:	(list of floats) Element i is the fidelity of the buffered
												link at the beginning of time slot i.
	- cons_requests:	(list of bools) Element i is True if there was a consumption
											request at the beginning of time slot i.
											In that case, the consumed fidelity is
											buffered_fidelity[i].
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
	buffered_fidelity = [None for i in range(t_end+1)]
	cons_requests = (np.random.rand(t_end+1) < p_cons)
	purif_events = []
	cons_fidelities = []
	cons_events = 0

	#------------------------------------------------------
	# Run process
	#------------------------------------------------------
	for t in range(t_end+1):
		buffered_fidelity[t] = F

		# Decohere
		if F is not None:
			F = 0.25 + (F-0.25)*np.exp(-Gamma)

		# Consume and go to next time slot
		if cons_requests[t] and F is not None:
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
				p_purif_succ, F_out = purif_protocol(F,rho_new,num_new_links)
				if np.random.rand() < p_purif_succ:
					# Success
					F = F_out
				else:
					# Failure
					F = None
				purif_events += [t]

	# Average consumed fidelity
	cons_fid_avg = np.mean(cons_fidelities)
	cons_fid_std = np.std(cons_fidelities)/len(cons_fidelities)

	# Availability 
	# 	(standard error for a Bernoulli process: https://en.wikipedia.org/wiki/
	# 	Binomial_proportion_confidence_interval#Standard_error_of_a_proportion_estimation_when_using_weighted_data)
	availability_avg = cons_events/sum(cons_requests)
	availability_stderr = np.sqrt(availability_avg*(1-availability_avg)/sum(cons_requests))

	return cons_fid_avg, cons_fid_std, availability_avg, availability_stderr, buffered_fidelity, cons_requests, purif_events



# def multiple_runs_1GnB(n, p_gen, rho_new, q_purif, purif_protocol, pur_after_swap, Gamma, p_cons, t_end, N_samples, randomseed):

# 	#------------------------------------------------------
# 	# Check valid parameter values
# 	#------------------------------------------------------
# 	assert p_gen>=0 and p_gen<=1
# 	assert q_purif>=0 and q_purif<=1
# 	assert p_cons>=0 and p_cons<=1

# 	if pur_after_swap:
# 		raise ValueError('pur_after_swap not implemented')
	
# 	#------------------------------------------------------
# 	# Initialize variables
# 	#------------------------------------------------------
# 	np.random.seed(randomseed)
# 	random.seed(randomseed)

# 	# Performance metrics
# 	availability_avg = [None for i in range(t_end+1)]
# 	availability_stderr = [None for i in range(t_end+1)]
# 	consumed_fidelity_avg = [None for i in range(t_end+1)]
# 	consumed_fidelity_stderr = [None for i in range(t_end+1)]
# 	rejections_avg = [None for i in range(t_end+1)]
# 	rejections_stderr = [None for i in range(t_end+1)]

# 	# Logs for 10 runs
# 	buffered_fidelities = []
# 	cons_requests_list = []
# 	purif_events_list = []

# 	#------------------------------------------------------
# 	# Initialize variables
# 	#------------------------------------------------------

# 	for sample in range(N_samples):
# 		buffered_fidelity, consumption_requests, purif_events = single_run_1GnB(n, p_gen, rho_new, q_purif, purif_protocol,
# 																	pur_after_swap, Gamma, p_cons, t_end, randomseed=None)
# 		if sample < 10:
# 			buffered_fidelities += [buffered_fidelity]
# 			cons_requests_list += [consumption_requests]
# 			purif_events_list += [purif_events]



# 	return availability_avg, availability_stderr,
# 		consumed_fidelity_avg, consumed_fidelity_stderr,
# 		rejections_avg, rejections_stderr,
# 		subset_single_runs


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------- PLOTS ----------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
def plot_run_1GnB(buffered_fidelity, cons_requests, purif_events, n, p_gen, rho_new, q_purif, purif_protocol, pur_after_swap, Gamma, p_cons, t_end, randomseed=None):
	"""
	Plot the fidelity of the buffered memory in the 1GnB entanglement buffer over time,
	for multiple realizations of the process.

	Parameters:
	- buffered_fidelity:	(list of floats) Element i is the fidelity of the buffered
												link at the beginning of time slot i.
	- cons_requests:	(list of bools) Element i is True if there was a consumption
											request at the beginning of time slot i.
											In that case, the consumed fidelity is
											buffered_fidelity[i].
	- purif_events:		(list of ints) Time slots in which purification was attempted.
	- n:		(int) Number of bad memories.
	- p_gen:	(float) Probability of successful entanglement generation
						in each bad memory at each time slot.
	- rho_new:	(np.array) Density matrix of newly generated entangled links,
							written in the Bell-state basis: 00+11, 00-11, 01+10, 01-10.
							The fidelity is the first entry of the matrix.
	- q_purif:	(float) Probability of purifying the link in memory when
						new links are generated.
	- purif_protocol:	(function) Returns the probability of success and the
									output fidelity of the purification protocol,
									for a given fidelity F of the buffered link,
									rho_new, and number of new links l.
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
	plt.scatter(range(t_end+1), buffered_fidelity)
	plt.plot(range(t_end+1), buffered_fidelity, color='tab:blue', marker='o', zorder=0)

	# Highlight consumption events
	cons_events = []
	F_cons_events = []
	for t, consumption in enumerate(cons_requests):
		if consumption:
			cons_events += [t]
			F_cons_events += [buffered_fidelity[t]]
	plt.scatter(cons_events, F_cons_events, color='k', marker='x', zorder=1, label='Consumption events')
	
	# Highlight purification events
	purif_events = purif_events
	plt.scatter(purif_events, [buffered_fidelity[t] for t in purif_events],
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













