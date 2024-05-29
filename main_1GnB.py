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
import functools

# Jupyter widgets
import ipywidgets
from ipywidgets import widgets
import pandas as pd
from IPython.display import clear_output
from IPython.display import display

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#-------------------------------- PURIFICATION POLICIES -----------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
def policy_label_to_function(policy_name):
	if policy_name == 'Identity':
		policy = policy_identity
	elif policy_name == 'Replacement':
		policy = policy_replacement
	elif policy_name == 'DEJMPS':
		policy = policy_DEJMPS
	elif policy_name == 'Double DEJMPS':
		policy = policy_doubleDEJMPS
	elif policy_name[0:16] == 'Concat. DEJMPS x':
		policy = functools.partial(policy_concatDEJMPS, max_links_used=int(policy_name[16:]))
	elif policy_name[0:15] == 'Nested DEJMPS x':
		policy = functools.partial(policy_concatDEJMPS, max_links_used=int(policy_name[15:]))
	elif policy_name == 'Optimal bi. Cliff.':
		policy = policy_opt_bilocal_Clifford
	elif policy_name == '313':
		policy = policy_313
	elif policy_name == '513 EC':
		policy = functools.partial(policy_513, mode='EC')
	elif policy_name == '513 ED':
		policy = functools.partial(policy_513, mode='ED')
	else:
		raise ValueError('Unknown policy')
	return policy

def policy_DEJMPS(rho_new, num_new_links):
	'''Purification policy:
		2-to-1: DEJMPS purification protocol.
		x-to-1: uses one new link for DEJMPS and discards the rest.

	Parameters:
	- rho_new:	(np.array) Density matrix of newly generated entangled links,
							written in the Bell-state basis: 00+11, 00-11, 01+10, 01-10.
							The fidelity is the first entry of the matrix.
	- num_new_links:	(int) Number of newly generated links. The protocol performs
								(num_new_links)-to-1 purification.

	Returns:
	- p_purif_succ:	(float) Probability of success.
	- F_out:	(float) Output fidelity.'''

	assert num_new_links >= 1

	# ## Werner state in memory ##
	# A_werner = F
	# B_werner = (1-F)/3
	# C_werner = (1-F)/3
	# D_werner = (1-F)/3

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
	a1 = (1/6) * (5*A-3*B+C+D) # Shifted output fidelity = (a1*(F-1/4) + b1) / (Prob of success)
	b1 = (1/24) * (3*A+5*B-3*C-3*D)

	return a1,b1,c1,d1

def policy_doubleDEJMPS(rho_new, num_new_links):
	'''Purification policy:
		2-to-1: DEJMPS purification protocol.
		x-to-1: uses one new link for DEJMPS and discards the rest.

	Parameters:
	- rho_new:	(np.array) Density matrix of newly generated entangled links,
							written in the Bell-state basis: 00+11, 00-11, 01+10, 01-10.
							The fidelity is the first entry of the matrix.
	- num_new_links:	(int) Number of newly generated links. The protocol performs
								(num_new_links)-to-1 purification.

	Returns:
	- p_purif_succ:	(float) Probability of success.
	- F_out:	(float) Output fidelity.'''

	assert num_new_links >= 1

	A_new = rho_new[0][0]
	B_new = rho_new[3][3]
	C_new = rho_new[2][2]
	D_new = rho_new[1][1]

	## Diagonal elements (in Bell-state basis) of the state that will be used to purify the ##
	## state stored in memory with DEJMPS ##
	A = rho_new[0][0]
	B = rho_new[3][3]
	C = rho_new[2][2]
	D = rho_new[1][1]

	## Do one application of DEJMPS with the new links ##
	p_success_newlinks = 1 # Probability of not failing any of these DEJMPS
	for ii in range(min(num_new_links-1,1)):
		# Probability of success of step ii
		p_success_round = (A+B)*(A_new+B_new) + (C+D)*(C_new+D_new)
		p_success_newlinks = p_success_newlinks*p_success_round
		# Output state of step ii
		A = (A*A_new+B*B_new)/p_success_round
		B = (C*D_new+D*C_new)/p_success_round
		C = (C*C_new+D*D_new)/p_success_round
		D = (A*B_new+B*A_new)/p_success_round

	## Use the resulting link to purify the link in memory ##
	## Purification coefficients ##
	c_l = (2/3) * (A+B-C-D) * p_success_newlinks
	d_l = (1/2) * (A+B+C+D) * p_success_newlinks
	a_l = (1/6) * (5*A-3*B+C+D) * p_success_newlinks
	b_l = (1/24) * (3*A+5*B-3*C-3*D) * p_success_newlinks

	return a_l,b_l,c_l,d_l

def policy_concatDEJMPS(rho_new, num_new_links, max_links_used=1):
	'''Purification policy:
		x-to-1: applies DEJMPS purification protocol at most max_links_used times.

	Parameters:
	- rho_new:	(np.array) Density matrix of newly generated entangled links,
							written in the Bell-state basis: 00+11, 00-11, 01+10, 01-10.
							The fidelity is the first entry of the matrix.
	- num_new_links:	(int) Number of newly generated links. The protocol performs
								(num_new_links)-to-1 purification.

	Returns:
	- p_purif_succ:	(float) Probability of success.
	- F_out:	(float) Output fidelity.'''

	assert num_new_links >= 1

	A_new = rho_new[0][0]
	B_new = rho_new[3][3]
	C_new = rho_new[2][2]
	D_new = rho_new[1][1]

	## Diagonal elements (in Bell-state basis) of the state that will be used to purify the ##
	## state stored in memory with DEJMPS ##
	A = rho_new[0][0]
	B = rho_new[3][3]
	C = rho_new[2][2]
	D = rho_new[1][1]

	## Do at most min(num_new_links-1, max_links_used-1) applications of DEJMPS with the new links ##
	p_success_newlinks = 1 # Probability of not failing any of these DEJMPS
	for ii in range(min(num_new_links-1, max_links_used-1)):
		# Probability of success of step ii
		p_success_round = (A+B)*(A_new+B_new) + (C+D)*(C_new+D_new)
		p_success_newlinks = p_success_newlinks*p_success_round
		# Output state of step ii
		A = (A*A_new+B*B_new)/p_success_round
		B = (C*D_new+D*C_new)/p_success_round
		C = (C*C_new+D*D_new)/p_success_round
		D = (A*B_new+B*A_new)/p_success_round

	## Use the resulting link to purify the link in memory ##
	## Purification coefficients ##
	c_l = (2/3) * (A+B-C-D) * p_success_newlinks
	d_l = (1/2) * (A+B+C+D) * p_success_newlinks
	a_l = (1/6) * (5*A-3*B+C+D) * p_success_newlinks
	b_l = (1/24) * (3*A+5*B-3*C-3*D) * p_success_newlinks

	return a_l,b_l,c_l,d_l

def policy_identity(rho_new, num_new_links):
	'''Purification policy:
		x-to-1: discards new links, keeps the one in memory.

	Parameters:
	- rho_new:	(np.array) Density matrix of newly generated entangled links,
							written in the Bell-state basis: 00+11, 00-11, 01+10, 01-10.
							The fidelity is the first entry of the matrix.
	- num_new_links:	(int) Number of newly generated links. The protocol performs
								(num_new_links)-to-1 purification.

	Returns:'''

	assert num_new_links >= 1

	#p_purif_succ = (A+B)*(A_werner+B_werner) + (C+D)*(C_werner+D_werner)
	#F_out = (A*A_werner + B*B_werner) / p_purif_succ

	## Purification coefficients ##
	c_l = 0 # Prob of success = c1*(F-1/4) + d1
	d_l = 1
	a_l = 1 # Shifted output fidelity = (a1*(F-1/4) + b1) / (Prob of success)
	b_l = 0

	return a_l,b_l,c_l,d_l

def policy_replacement(rho_new, num_new_links):
	'''Purification policy:
		x-to-1: replaces the link in memory with a fresh one.

	Parameters:
	- rho_new:	(np.array) Density matrix of newly generated entangled links,
							written in the Bell-state basis: 00+11, 00-11, 01+10, 01-10.
							The fidelity is the first entry of the matrix.
	- num_new_links:	(int) Number of newly generated links. The protocol performs
								(num_new_links)-to-1 purification.

	Returns:'''

	assert num_new_links >= 1

	#p_purif_succ = (A+B)*(A_werner+B_werner) + (C+D)*(C_werner+D_werner)
	#F_out = (A*A_werner + B*B_werner) / p_purif_succ

	## Purification coefficients ##
	c_l = 0 # Prob of success = c1*(F-1/4) + d1
	d_l = 1
	a_l = 0 # Shifted output fidelity = (a1*(F-1/4) + b1) / (Prob of success)
	b_l = rho_new[0][0]-1/4

	return a_l,b_l,c_l,d_l

def policy_opt_bilocal_Clifford(rho_new, num_new_links):
	'''Purification policy:
		num_new_links=1: DEJMPS purification protocol.
		num_new_links>1: Apply the optimal bilocal Clifford protocol from Appendix E
						from Jansen2022 on the new links (only applies to Werner states).
						Then use the remaining link to do DEJMPS on the buffered
						link. If num_new_links>8, it only uses 8 new links
						and discards the rest.

	Parameters:
	- rho_new:	(np.array) Density matrix of newly generated entangled links,
							written in the Bell-state basis: 00+11, 00-11, 01+10, 01-10.
							The fidelity is the first entry of the matrix.
	- num_new_links:	(int) Number of newly generated links. The protocol performs
								(num_new_links)-to-1 purification.

	Returns: ...'''

	assert num_new_links >= 1
	
	## The Jansen policy only works for Werner states ##
	assert isWerner(rho_new), 'Optimal bilocal Clifford policy is only defined for Werner states!'

	## Diagonal elements of the newly generated state (in Bell-state basis) ##
	A_new = rho_new[0][0]
	B_new = rho_new[3][3]
	C_new = rho_new[2][2]
	D_new = rho_new[1][1]

	## Compute the probability of not failing the purification of new links ##
	A = A_new
	if num_new_links==1:
		p_success_newlinks = 1
	elif num_new_links==2:
		p_success_newlinks = (8/9)*A**2 - (4/9)*A + 5/9
	elif num_new_links==3:
		p_success_newlinks = (32/27)*A**3 - (4/9)*A**2 + 7/27
	elif num_new_links==4:
		p_success_newlinks = (32/27)*A**4 - (4/9)*A**2 + (4/27)*A + 1/9
	elif num_new_links==5:
		p_success_newlinks = (80/27)*A**4 - (80/27)*A**3 + (10/9)*A**2 - (5/27)*A + 2/27
	elif num_new_links==6:
		p_success_newlinks = (128/243)*A**6 + (320/243)*A**5 - (256/243)*A**4 + (16/243)*A**3 + (40/243)*A**2 - (14/243)*A + 1/27
	elif num_new_links==7:
		p_success_newlinks = (2048/2187)*A**7 - (128/2187)*A**6 + (320/729)*A**5 - (796/2187)*A**4 - (44/2187)*A**3 + (49/729)*A**2 - (37/2187)*A + 37/2187
	elif num_new_links==8:
		p_success_newlinks = (6656/6561)*A**8 - (1024/6561)*A**7 + (1664/6561)*A**6 - (64/6561)*A**5 - (1120/6561)*A**4 + (416/6561)*A**3 - (4/6561)*A**2 - (16/6561)*A + 53/6561
	else:
		raise ValueError('Optimal bilocal Clifford policy only works up to num_new_links=8')


	## And the fidelity of the state after num_new_links-to-1 purifications ##
	if num_new_links==1:
		A = A_new
	elif num_new_links==2:
		A = ( (10/9)*A**2 - (2/9)*A + 1/9 )/p_success_newlinks
	elif num_new_links==3:
		A = ( (28/27)*A**3 - (1/9)*A + 2/27 )/p_success_newlinks
	elif num_new_links==4:
		A = ( (8/9)*A**4 + (8/27)*A**3 - (2/9)*A**2 + 1/27 )/p_success_newlinks
	elif num_new_links==5:
		A = ( (32/27)*A**5 - (20/27)*A**4 + (10/9)*A**3 - (20/27)*A**2 + (5/27)*A )/p_success_newlinks
	elif num_new_links==6:
		A = ( (32/27)*A**6 - (112/243)*A**5 + (80/243)*A**4 + (8/243)*A**3 - (32/243)*A**2 + (10/243)*A + 1/243 )/p_success_newlinks
	elif num_new_links==7:
		A = ( (2368/2187)*A**7 - (592/2187)*A**6 + (196/729)*A**5 - (44/2187)*A**4 - (199/2187)*A**3 + (20/729)*A**2 - (2/2187)*A + 8/2187 )/p_success_newlinks
	elif num_new_links==8:
		raise ValueError('The F_out reported in Jansen2022 for num_new_links=8 is incorrect')
		A = ( (6784/6561)*A**8 - (51/6561)*A**7 - (32/6561)*A**6 + (832/6561)*A**5 - (560/6561)*A**4 - (8/6561)*A**3 + (52/6561)*A**2 - (8/6561)*A + 13/6561 )/p_success_newlinks
	else:
		raise ValueError('Optimal bilocal Clifford policy only works up to num_new_links=8')

	## Since the state is Werner, the other diagonal elements are trivial to compute ##
	B = (1-A)/3
	C = (1-A)/3
	D = (1-A)/3

	## Use the resulting link to purify the link in memory ##
	## Purification coefficients ##
	c_l = (2/3) * (A+B-C-D) * p_success_newlinks
	d_l = (1/2) * (A+B+C+D) * p_success_newlinks
	a_l = (1/6) * (5*A-3*B+C+D) * p_success_newlinks
	b_l = (1/24) * (3*A+5*B-3*C-3*D) * p_success_newlinks

	return a_l,b_l,c_l,d_l

def policy_313(rho_new, num_new_links):
	'''Purification policy:
		num_new_links<3: Apply DEJMPS.
		num_new_links>=3: Apply the deterministic purification protocol based on the
							[[3,1,3]] convolutional code used in Patil2024 to 
							3 new links. Then, twirl to make Werner
							(because I'm unsure how the output states from 313 protocol
							look like) and apply DEJMPS to the buffered link. The other
							links are discarded.

	Parameters:
	- rho_new:	(np.array) Density matrix of newly generated entangled links,
							written in the Bell-state basis: 00+11, 00-11, 01+10, 01-10.
							The fidelity is the first entry of the matrix.
	- num_new_links:	(int) Number of newly generated links. The protocol performs
								(num_new_links)-to-1 purification.

	Returns: ...'''

	assert num_new_links >= 1

	if num_new_links < 3:
		return policy_DEJMPS(rho_new, 1)
	
	## The 313 policy only works for Werner states ##
	assert isWerner(rho_new), '313 policy is only defined for Werner states!'

	## Diagonal elements of the newly generated state (in Bell-state basis) ##
	A_new = rho_new[0][0]
	B_new = rho_new[3][3]
	C_new = rho_new[2][2]
	D_new = rho_new[1][1]

	## Compute the probability of not failing the purification of new links ##
	p_success_newlinks = 1

	## Fidelity of the 313-purified links ##
	z = 1 - 0.998 * A_new
	A = 1 - (0.0013423 - 0.465895*z + 92.816480*z**2 - 600.2412*z**3 + 1141.6515*z**4)
	# This function was obtained by fitting fourth order polynomial (with WebPlotDigitizer
	# and Plotly) to the blue curve from Figure 1 from Patil2024, and then tweaked (rescaled
	# infidelity 1-A_new to z) to create a fixed point

	## We twirl the output state into Werner form ##
	B = (1-A)/3
	C = (1-A)/3
	D = (1-A)/3

	## Use the resulting link to purify the link in memory ##
	## Purification coefficients ##
	c_l = (2/3) * (A+B-C-D) * p_success_newlinks
	d_l = (1/2) * (A+B+C+D) * p_success_newlinks
	a_l = (1/6) * (5*A-3*B+C+D) * p_success_newlinks
	b_l = (1/24) * (3*A+5*B-3*C-3*D) * p_success_newlinks

	return a_l,b_l,c_l,d_l

def policy_513(rho_new, num_new_links, mode='EC'):
	'''Purification policy:
		num_new_links<5 or F_new<0.86: Apply double DEJMPS (since 513 EC/ED only works well
										for fidelities larger than 0.86/0.75 approx)
		else: Apply the deterministic purification protocol based on the
							[[5,1,3]] code in error-correcting (EC) or error-detecting (ED)
							mode, used in Stephens2013, to 5 new links.
							Then, twirl to make Werner (because I'm unsure how the output
							state looks like) and apply DEJMPS to the buffered link.
							The other links are discarded.

	Parameters:
	- rho_new:	(np.array) Density matrix of newly generated entangled links,
							written in the Bell-state basis: 00+11, 00-11, 01+10, 01-10.
							The fidelity is the first entry of the matrix.
	- num_new_links:	(int) Number of newly generated links. The protocol performs
								(num_new_links)-to-1 purification.

	Returns: ...'''

	assert num_new_links >= 1

	if num_new_links < 5 or (mode=='EC'and rho_new[0][0] < 0.86) or (mode=='ED'and rho_new[0][0] < 0.75):
		return policy_concatDEJMPS(rho_new, num_new_links, max_links_used=2)
	
	## The 513 policy only works for Werner states ##
	assert isWerner(rho_new), '513 policy is only defined for Werner states!'

	## Diagonal elements of the newly generated state (in Bell-state basis) ##
	A_new = rho_new[0][0]
	B_new = rho_new[3][3]
	C_new = rho_new[2][2]
	D_new = rho_new[1][1]

	## Compute the probability of not failing the purification of new links ##
	if mode == 'EC':
		p_success_newlinks = 0.404 + 0.596 * np.exp( -(A_new-1)**2 / (2*0.199**2) )
	elif mode == 'ED':
		p_success_newlinks = -0.153 + 0.01488 * np.exp( 4.34976 * A_new )

	## Fidelity of the 313-purified links ##
	if mode == 'EC':
		A = 0.669 + 0.331 * np.exp( -(A_new-1)**2 / (2*0.136**2) )
	elif mode == 'ED':
		A = 1 - 0.046 * np.exp( -(A_new-0.680)**2 / (2*0.083**2) )

	## We twirl the output state into Werner form ##
	B = (1-A)/3
	C = (1-A)/3
	D = (1-A)/3

	## Use the resulting link to purify the link in memory ##
	## Purification coefficients ##
	c_l = (2/3) * (A+B-C-D) * p_success_newlinks
	d_l = (1/2) * (A+B+C+D) * p_success_newlinks
	a_l = (1/6) * (5*A-3*B+C+D) * p_success_newlinks
	b_l = (1/24) * (3*A+5*B-3*C-3*D) * p_success_newlinks

	return a_l,b_l,c_l,d_l


#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------- SIMULATION -----------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def single_run_1GnB(n, p_gen, rho_new, q_purif, purif_policy, pur_after_swap, Gamma, p_cons, t_end, randomseed, burn=None):
	'''Simulates the 1GnB entanglement buffer.
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
	- purif_events:		(list of ints) Time slots in which purification was attempted.'''

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
		# Decohere
		if F is not None:
			F = 0.25 + (F-0.25)*np.exp(-Gamma)

		buffered_fidelity_trace[t] = F

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
				a_l,b_l,c_l,d_l = purif_policy(rho_new,num_new_links)
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
#------------------------------------- ANALYTICS ------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def isWerner(rho):
	'''Checks if a two-qubit density matrix is a Werner state'''
	result = True * (rho[1][1] == rho[2][2]) * (rho[1][1] == rho[3][3])
	rho_zeros = (rho==0)
	for i in range(len(rho)):
		for j in range(len(rho)):
			if i != j:
				result *= rho_zeros[i][j]
	return result

def analytical_availability_Fcons(n, p_gen, rho_new, q_purif, purif_policy, pur_after_swap, Gamma, p_cons):
	'''Computes the availability and the average consumed fidelity (Fcons)
		using our analytical solution.

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

	Returns:
	- A:	(float) Availability (ratio of accepted consumption requests).'''


	## Compute purification constants ##
	purif_constants = [purif_policy(rho_new, l) for l in range(1,n+1)]

	## Tilde constants ##
	a_tilde = sum([purif_constants[l-1][0]*math.comb(n,l)*(1-p_gen)**(n-l)*p_gen**l for l in range(1,n+1)])
	b_tilde = sum([purif_constants[l-1][1]*math.comb(n,l)*(1-p_gen)**(n-l)*p_gen**l for l in range(1,n+1)])
	c_tilde = sum([purif_constants[l-1][2]*math.comb(n,l)*(1-p_gen)**(n-l)*p_gen**l for l in range(1,n+1)])
	d_tilde = sum([purif_constants[l-1][3]*math.comb(n,l)*(1-p_gen)**(n-l)*p_gen**l for l in range(1,n+1)])

	## Big Tilde constants ##
	A_tilde = (q_purif * np.exp(-Gamma)*(1-p_cons)*a_tilde) / (1 - np.exp(-Gamma) * (1-q_purif+q_purif*(1-p_gen)**n) * (1-p_cons))
	B_tilde = (q_purif * (1-p_cons)*b_tilde) / ( p_cons + q_purif*(1-(1-p_gen)**n)*(1-p_cons) )
	C_tilde = (q_purif * np.exp(-Gamma)*(1-p_cons)*c_tilde) / (1 - np.exp(-Gamma) * (1-q_purif+q_purif*(1-p_gen)**n) * (1-p_cons))
	D_tilde = (q_purif * (1-p_cons)*d_tilde) / ( p_cons + q_purif*(1-(1-p_gen)**n)*(1-p_cons) )

	## Intermediate variables ##
	g_new = rho_new[0][0] - 1/4
	y = ( B_tilde*C_tilde + C_tilde*g_new + D_tilde*(1-A_tilde) ) / ((1-A_tilde)*(1-D_tilde) - B_tilde*C_tilde)
	x = ( A_tilde*g_new*(1-D_tilde) + B_tilde + B_tilde*C_tilde*g_new ) / ( (1-A_tilde)*(1-D_tilde) - B_tilde*C_tilde )
	expected_T_N = (1+y) / ( p_cons + q_purif * (1-(1-p_gen)**n) * (1-p_cons) )
	expected_T_gen = 1 / ( 1 - (1-p_gen)**n )

	## Availability and Fcons ##
	A = expected_T_N / (expected_T_N + expected_T_gen)
	Fcons = 1/4 + (g_new+x)*( p_cons + q_purif * (1-(1-p_gen)**n) * (1-p_cons) ) / ( (1+y)
						* (np.exp(Gamma) - 1 + p_cons + q_purif * (1-(1-p_gen)**n) * (1-p_cons) ) )

	return A, Fcons

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

def AFplot(policy_names, sim_data=None, theory_data=None, filename=None, xlims=None, ylims=None):
	fig, ax = plt.subplots()

	if False: # filename:
		figsize_cm = 9
	else:
		figsize_cm = 20
	fig.set_size_inches(figsize_cm/2.54, figsize_cm/2.54)
	
	## Colors and markers ##
	#colors = ['k', 'tab:blue', 'tab:orange', 'tab:purple']
	cmap = plt.cm.get_cmap('inferno')
	colors = [cmap(i/len(policy_names)) for i in range(len(policy_names))]
	markers = ['^','v','o','s','d']

	## Plot simulation data ##
	if sim_data:
		for idx_policy, policy in enumerate(policy_names):
			plt.errorbar(sim_data['A_avg_vec'][idx_policy], sim_data['Fcons_avg_vec'][idx_policy],
						 xerr=sim_data['A_stderr_vec'][idx_policy],
						 yerr=sim_data['Fcons_stderr_vec'][idx_policy],
						 marker=markers[idx_policy], color=colors[idx_policy],
						 linestyle=':', capsize=3, label=policy+' (sim.)')
	
	## Plot theory data ##
	if theory_data:
		for idx_policy, policy in enumerate(policy_names):
			plt.plot(theory_data['A'][idx_policy], theory_data['Fcons_avg'][idx_policy],
						 linewidth=3,
						 color=colors[idx_policy], linestyle='-', label=policy)#+' (theory)')
		
	
	## Plot specs ##
	plt.legend(fontsize=14)
	plt.xlabel(r'Availability')
	plt.ylabel(r'Avg. consumed fidelity')
	ax.tick_params(labelsize=14) # Font size

	dA = 0.05
	if xlims==None:
		xmin = round(np.floor(np.min(theory_data['A']) / dA) * dA,2)
		xmax = round(np.ceil(np.max(theory_data['A']) / dA) * dA,2)
	else:
		xmin = xlims[0]
		xmax = xlims[1]
	dF = 0.05
	if ylims==None:
		ymin = round(np.floor(np.min(theory_data['Fcons_avg']) / dF) * dF,2)
		ymax = round(np.ceil(np.max(theory_data['Fcons_avg']) / dF) * dF,2)
	else:
		ymin = ylims[0]
		ymax = ylims[1]
	plt.xlim(xmin, xmax)
	plt.ylim(ymin, ymax)
	ax.set_xticks(np.arange(xmin,xmax*1.0001,dA))
	ax.set_yticks(np.arange(ymin,ymax*1.0001,dF))
	
	if filename:
		plt.savefig(filename, dpi=300, bbox_inches='tight')
	else:
		plt.show()
	return

def AFplot_theory(varying_param, n, p_gen, rho_new, q_purif, policy_names, pur_after_swap, Gamma, p_cons, savefig=False, xlims=None, ylims=None):
	if varying_param=='q_purif':
		varying_array = q_purif
	else:
		raise ValueError('Unknown varying_param')

	## COMPUTE THEORY ##
	Fcons_theory_vec = [[] for policy in policy_names]
	A_theory_vec = [[] for policy in policy_names]

	for idx_policy, policy_name in enumerate(policy_names):
		purif_policy = policy_label_to_function(policy_name)
		for x in varying_array:
			if varying_param=='q_purif':
				q_purif = x
			else:
				raise ValueError('Unknown varying_param')
			A, Fcons = analytical_availability_Fcons(n, p_gen, rho_new, q_purif, purif_policy, pur_after_swap, Gamma, p_cons)
			Fcons_theory_vec[idx_policy] += [Fcons]
			A_theory_vec[idx_policy] += [A]
	theory_data = {'Fcons_avg': Fcons_theory_vec, 'A': A_theory_vec}

	## PLOT ##
	if savefig:
		filename = 'figs/AF_theory_%s'%varying_param
		if varying_param=='q_purif':
			filename += '_n%d_pgen%.3f_pcons%.3f_rhodiag-%.3f-%.3f-%.3f-%.3f_swapandpur%s_G%.5f.pdf'%(n, p_gen, p_cons,
							rho_new[0][0], rho_new[1][1], rho_new[2][2], rho_new[3][3], pur_after_swap, Gamma)
		else:
			raise ValueError('Unknown varying_param')
	else:
		filename = None
	AFplot(policy_names, sim_data=None, theory_data=theory_data, filename=filename, xlims=xlims, ylims=xlims)

	return

def AFplot_interactive(policy_names):
	## Widgets specs ##
	slider_layout = ipywidgets.Layout(width='60%')

	## Plot specs ##
	xlims = [0.5,1]
	ylims = [0.5,1]



	def AFplot_simple_inputs(n, p_gen, new_states, gen_tradeoff_parameter, rho00, rho11, rho22, Gamma, p_cons, policy_names):
		if new_states == 'Werner':
			rho_new = np.diag([rho00, (1-rho00)/3, (1-rho00)/3, (1-rho00)/3])
		elif new_states == 'Werner tradeoff':
			rho00 = 1-gen_tradeoff_parameter*p_gen
			rho_new = np.diag([rho00, (1-rho00)/3, (1-rho00)/3, (1-rho00)/3])
		elif new_states == 'Bell-diagonal':
			if rho00+rho11+rho22 > 1:
				print('Invalid density matrix')
				return
			rho_new = np.diag([rho00, rho11, rho22, 1-rho00-rho11-rho22])
		elif new_states == 'R-state':
			rho_new = np.diag([rho00, 0, (1-rho00)/2, (1-rho00)/2])

		pur_after_swap = False
		varying_param = 'q_purif'
		q_purif = np.linspace(0,1,10)

		AFplot_theory(varying_param, n, p_gen, rho_new, q_purif, policy_names, pur_after_swap, Gamma, p_cons,
						savefig=False, xlims=xlims, ylims=ylims)
		return

	new_states_widget = widgets.RadioButtons(options=['Werner', 'Werner tradeoff', 'Bell-diagonal', 'R-state'], value='Werner', description=r'$\rho_\mathrm{new}$')
	rho00_widget = widgets.FloatSlider(value=0.9, min=0.5, max=1, step=0.05, description=r'$F_\mathrm{new}$', layout=slider_layout)
	rho11_widget = widgets.FloatSlider(value=0.033, min=0, max=0.1, step=0.001, description=r'$\rho_\mathrm{new,11}$', disabled=True, layout=slider_layout)
	rho22_widget = widgets.FloatSlider(value=0.033, min=0, max=0.1, step=0.001, description=r'$\rho_\mathrm{new,22}$', disabled=True, layout=slider_layout)
	gen_tradeoff_widget = widgets.FloatSlider(value=0.5, min=0, max=1, step=0.1, description=r'$\lambda_\mathrm{gen}$', disabled=True, layout=slider_layout)
	def update_rho_new_widgets(*args):
		if new_states_widget.value in ['Werner', 'Werner tradeoff', 'R-state']:
			rho11_widget.disabled = True
			rho22_widget.disabled = True
		else:
			rho11_widget.disabled = False
			rho22_widget.disabled = False
		if new_states_widget.value in ['Werner tradeoff']:
			gen_tradeoff_widget.disabled = False
			rho00_widget.disabled = True
		else:
			gen_tradeoff_widget.disabled = True
			rho00_widget.disabled = False
		rho11_widget.max = 1-rho00_widget.value
		rho22_widget.max = 1-rho00_widget.value-rho11_widget.value
	new_states_widget.observe(update_rho_new_widgets, 'value')
	rho00_widget.observe(update_rho_new_widgets, 'value')
	rho11_widget.observe(update_rho_new_widgets, 'value')

	ipywidgets.interact(AFplot_simple_inputs,
		n = widgets.IntSlider(value=1, min=1, max=15, step=1, layout=slider_layout),
		p_gen = widgets.FloatSlider(value=0.5, min=0, max=1, step=0.05, description=r'$p_\mathrm{gen}$', layout=slider_layout),
		new_states = new_states_widget,
		gen_tradeoff_parameter = gen_tradeoff_widget,
		rho00 = rho00_widget,
		rho11 = rho11_widget,
		rho22 = rho22_widget,
		Gamma = widgets.FloatSlider(value=0.2, min=0, max=1, step=0.01, description=r'$\Gamma$', layout=slider_layout),# continuous_update=False),
		p_cons = widgets.FloatSlider(value=0.1, min=0, max=0.3, step=0.01, description=r'$p_\mathrm{cons}$', layout=slider_layout),
		policy_names = widgets.SelectMultiple(options=policy_names, value=policy_names, rows=len(policy_names), description='Display'))
	return

def policies_plot(policy_names, rho_new, num_new_links, figsize_cm=20, savefig=False):

	F_vec = np.linspace(0.5,1,20)

	## CALCULATIONS ##
	p_success_vec = [[] for policy in policy_names]
	F_out_vec = [[] for policy in policy_names]

	for idx_policy, policy_name in enumerate(policy_names):
		purif_policy = policy_label_to_function(policy_name)
		a,b,c,d = purif_policy(rho_new, num_new_links)
		p_success_vec[idx_policy] = [c*(f-1/4)+d for f in F_vec]
		F_out_vec[idx_policy] = [1/4 + (a*(f-1/4)+b)/(c*(f-1/4)+d) for f in F_vec]

	## PLOT ##
	fig, ax = plt.subplots()
	fig.set_size_inches(figsize_cm/2.54, figsize_cm/2.54)
	
	# Colors and markers
	cmap = plt.cm.get_cmap('inferno')
	colors = [cmap(i/len(policy_names)) for i in range(len(policy_names))]
	markers = ['^','v','o','s','d']
	
	# Plot data
	plt.plot(F_vec,F_vec,color='k',linestyle=':',label='Reference (y=x)')
	for idx_policy, policy in enumerate(policy_names):
		plt.plot(F_vec, F_out_vec[idx_policy],
				 color=colors[idx_policy], linestyle='-', label=policy+r' ($F_\mathrm{out}$)')
	for idx_policy, policy in enumerate(policy_names):
		plt.plot(F_vec, p_success_vec[idx_policy],
				 color=colors[idx_policy], linestyle='--', label=policy+r' ($p$)')
	#plt.plot(F_vec,[3*f/(2*f+1) for f in F_vec], ':', color='tab:orange', label='Fout DEJMPS a mano')
	#plt.plot(F_vec,(2*F_vec+1)/3, ':', color='tab:blue', label='p DEJMPS a mano')

	# Plot specs
	plt.legend()
	plt.xlabel(r'Input fidelity (Werner state)')
	plt.ylabel(r'Output fidelity and success probability')

	plt.xlim(0.5, 1)
	plt.ylim(0, 1.01)
	ax.set_xticks([0.5,0.6,0.7,0.8,0.9,1])
	ax.set_yticks([0,0.25,0.5,0.75,1])

	if savefig:
		raise ValueError('Save not implemented!')
	else:
		plt.show()

	return

def policies_plot_interactive(policy_names):
	## Widgets specs ##
	slider_layout = ipywidgets.Layout(width='60%')

	def policies_plot_inputs(num_new_links, new_states, gen_tradeoff_parameter, rho00, policy_names):
		if new_states == 'Werner':
			rho_new = np.diag([rho00, (1-rho00)/3, (1-rho00)/3, (1-rho00)/3])
		policies_plot(policy_names, rho_new, num_new_links)
		return

	new_states_widget = widgets.RadioButtons(options=['Werner', 'Werner tradeoff'], value='Werner', description=r'$\rho_\mathrm{new}$')
	rho00_widget = widgets.FloatSlider(value=0.9, min=0.5, max=1, step=0.01, description=r'$F_\mathrm{new}$', layout=slider_layout)
	gen_tradeoff_widget = widgets.FloatSlider(value=0.5, min=0, max=1, step=0.1, description=r'$\lambda_\mathrm{gen}$', disabled=True, layout=slider_layout)
	def update_rho_new_widgets(*args):
		if new_states_widget.value in ['Werner tradeoff']:
			gen_tradeoff_widget.disabled = False
			rho00_widget.disabled = True
		else:
			gen_tradeoff_widget.disabled = True
			rho00_widget.disabled = False
	new_states_widget.observe(update_rho_new_widgets, 'value')
	rho00_widget.observe(update_rho_new_widgets, 'value')

	ipywidgets.interact(policies_plot_inputs,
		num_new_links = widgets.IntSlider(value=1, min=1, max=15, step=1, layout=slider_layout),
		new_states = new_states_widget,
		gen_tradeoff_parameter = gen_tradeoff_widget,
		rho00 = rho00_widget,
		policy_names = widgets.SelectMultiple(options=policy_names, value=policy_names, rows=len(policy_names), description='Display'))
	return

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------





























