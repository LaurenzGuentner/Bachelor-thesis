# This file contains all the functions that are being used in the main notebook.

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft,fftfreq,fftshift
import numba
from numba import njit, prange
from tqdm import tqdm

# Here I'm refactoring my two_state_process function into an optimized vectorized version that is also parallelizable

# The perturbation function that is called inside the generate_markov_jump_process function must be defined outside since numba can't handle nestef functions
@njit
def p(t,omega_p,epsilon):
    return epsilon * np.cos(omega_p*t)

@njit
def generate_markov_jump_process(number_of_substates: list, transition_rates: list, refractory_periods: list, total_time: float, delta_t: float, out = None, enable_perturbation = False, omega_p = 1.0, epsilon = 0.0):
    """Function creates a markov jump process that can be used to approximate
        a two state renewal process with refractory period.
        The outputs are two numpy arrays:
         - time_sequence containing a discrete time array up to total time with evenly spaced time steps delta t
         - state_sequence which contains the corresponding occupied state at each time step. The associated value is the state index from 0 to N+M and has to be mapped by a different function
        The other inputs are:
         - numpy array containing the number of substates N of chain A and the number of substates in M of chain B
         - numpy array of transition rates of the last state of each chain to the corresponding other chain
         - numpy array of refractory periods for the two chains that are aproximated by the N-1 / M-1 first substates of each chain
         - numpy array out which can be preallocated for efficiency in the use case of huge number of iterated calls of the function

        Noticably this function is optimized for parallel processing with Numba.
    """

    ### FLAGS ###
    transition_rates_substates_A_zero_flag = False
    transition_rates_substates_B_zero_flag = False
    no_states_in_chain_A_flag = False
    no_states_in_chain_B_flag = False
    #############


    N_A = number_of_substates[0]
    N_B = number_of_substates[1]
    
    target_sequence_length = int(np.floor(total_time/delta_t)) + 1
    sampling_rate = 1 / delta_t

    initial_state = 0
    current_state = initial_state

    # Define current time variable needed if we want to include a perturbation
    current_time = 0.0

    # Define position index in sequence
    pos = 0 

    # If no preallocated output array is provided create a new output array
    if out == None:
        out = np.empty(target_sequence_length)

    # Define transition rates of substates and handle divide by zero
    if refractory_periods[0] > 0.0:
        transition_rate_substates_A = (N_A - 1)/refractory_periods[0]    # in both cases for N substates we have N-1 transitions that approximate the refractory period
    else: 
        transition_rate_substates_A = 10**6
        print('Warning: The refactory period of chain A is zero!')
    if refractory_periods[1] > 0.0:  
        transition_rate_substates_B = (N_B-1)/refractory_periods[1]
    else: 
        transition_rate_substates_B = 10**6
        print('Warning: The refactory period of chain B is zero!')

    # Define static transition rate of last state of chain A and B
    transition_rate_A = transition_rates[0]
    transition_rate_B = transition_rates[1]

    # Now we fill the out array that holds the state sequence successively 
    while pos < target_sequence_length:
        # go through CHAIN A
        # first N_A-1 states approximate refractory period
        if N_A - 1 >= 0.0:
            if transition_rate_substates_A > 0.0:
                for substate_i in range(N_A - 1):
                    u = np.random.random()                  # This function generates uniformly distributed random numbers in [0,1). This method is used because rng.exponential is not supported in numba parallelization
                    while u == 0.0:                           # Avoid u = 0 since ln(0) = -infty
                        u = np.random.random()              # So we regenerate until u not equal to 0
                    residency_time = - np.log(u) / transition_rate_substates_A          # Transform uniformly sampled random variable into exponentially sampled time
                    # If we only floored the number of samples here we would introduce bias and make all residency times a little smaller
                    # Thus we probabilitsically ceil sometimes the rate of which is proportional to the difference between float and floored number of samples
                    frac = residency_time*sampling_rate - np.floor(residency_time*sampling_rate)
                    n_occupancy = max(1, int(np.floor(residency_time*sampling_rate) + (np.random.random() < frac)))        # We always at least spend one time step in each state even if n_occ would actually be less than 0
                    # Add vals to output array
                    end_pos = pos + n_occupancy
                    if end_pos > target_sequence_length:
                        end_pos = target_sequence_length
                    for k in range(pos,end_pos):
                        out[k] = current_state
                    # Update position index and state index
                    pos = end_pos
                    current_state += 1
                    current_time += n_occupancy * delta_t
                    # Break if state sequence is completely filled
                    if pos >= target_sequence_length:
                        break
            else:                                                           # Handle exceptions
                transition_rates_substates_A_zero_flag = True
                continue
        else: 
            no_states_in_chain_A_flag = True
            continue

        # Break if state sequence is completely filled after refractory substates
        if pos >= target_sequence_length:
            break

        # Handle last state of chain A seperatly
        if transition_rates[0] > 0.0:
            if enable_perturbation == False:
                u = np.random.random()
                while u == 0.0:                           
                    u = np.random.random()
                residency_time = - np.log(u) / transition_rate_A
            elif enable_perturbation == True:
                # Thinning algorithm
                accept = False
                tau = 0
                lambda_max = transition_rates[0]+epsilon
                while accept == False:
                    u = np.random.random()
                    while u == 0.0:                           
                        u = np.random.random()
                    # Sample first residency candidate
                    residency_time = - np.log(u) / lambda_max
                    tau += residency_time
                    current_transition_rate = transition_rates[0] + p(current_time + tau,omega_p,epsilon)
                    w = np.random.random()
                    if w <= current_transition_rate/(lambda_max):
                        accept = True
                        residency_time = tau
                    




            frac = residency_time*sampling_rate - np.floor(residency_time*sampling_rate)
            n_occupancy = max(1, int(np.floor(residency_time*sampling_rate) + (np.random.random() < frac))) 
            end_pos = pos + n_occupancy
            if end_pos > target_sequence_length:
                end_pos = target_sequence_length
            for k in range(pos,end_pos):
                out[k] = current_state
            # Update position index and state index
            pos = end_pos
            current_state += 1
            current_time += n_occupancy * delta_t
        else:           # If the transition rate is zero we stay in this state forever
            for k in range(pos,target_sequence_length):
                out[k] = current_state
            pos = target_sequence_length

        # Break if state sequence is completely filled
        if pos >= target_sequence_length:
            break

        ######

        # Move on to CHAIN B
        if N_B - 1 >= 0.0:
            if transition_rate_substates_B > 0.0:
                for substate_i in range(N_B - 1):
                    u = np.random.random()                  
                    while u == 0.0:                           
                        u = np.random.random()              
                    residency_time = - np.log(u) / transition_rate_substates_B         
                    frac = residency_time*sampling_rate - np.floor(residency_time*sampling_rate)
                    n_occupancy = max(1, int(np.floor(residency_time*sampling_rate) + (np.random.random() < frac)))     
                    # Add vals to output array
                    end_pos = pos + n_occupancy
                    if end_pos > target_sequence_length:
                        end_pos = target_sequence_length
                    for k in range(pos,end_pos):
                        out[k] = current_state
                    # Update position index and state index
                    pos = end_pos
                    current_state += 1
                    current_time += n_occupancy * delta_t
                    # Break if state sequence is completely filled
                    if pos >= target_sequence_length:
                        break
            else:                                                           # Handle exceptions
                transition_rates_substates_B_zero_flag = True
                continue
        else: 
            no_states_in_chain_B_flag = True
            continue

        # Break if state sequence is completely filled after refractory substates
        if pos >= target_sequence_length:
            break

        # Handle last state of chain B seperatly
        if transition_rates[1] > 0.0:
            u = np.random.random()
            while u == 0.0:                           
                u = np.random.random()
            residency_time = - np.log(u) / transition_rate_B
            frac = residency_time*sampling_rate - np.floor(residency_time*sampling_rate)
            n_occupancy = max(1, int(np.floor(residency_time*sampling_rate) + (np.random.random() < frac))) 
            end_pos = pos + n_occupancy
            if end_pos > target_sequence_length:
                end_pos = target_sequence_length
            for k in range(pos,end_pos):
                out[k] = current_state
            # Update position index and state index
            pos = end_pos
            current_state = 0  # Return to first state of chain A
            current_time += n_occupancy * delta_t
        else:           # If the transition rate is zero we stay in this state forever
            for k in range(pos,target_sequence_length):
                out[k] = current_state
            pos = target_sequence_length

        # Break if state sequence is completely filled
        if pos >= target_sequence_length:
            break

    # Flags
    if transition_rates_substates_A_zero_flag == True:
        print('Warning: The transition rates of the substates in chain A are zero. Are there no substates?')
    if no_states_in_chain_A_flag == True:
        print('Warning: There are no states in chain A!')
    if transition_rates_substates_B_zero_flag == True:
        print('Warning: The transition rates of the substates in chain B are zero. Are there no substates?')
    if no_states_in_chain_B_flag == True:
        print('Warning: There are no states in chain B!')

    # Generate time sequence
    time_sequence = np.arange(target_sequence_length) * delta_t
    # Clearify output
    state_sequence = out

    return time_sequence, state_sequence

@njit
def state_mapping(number_of_substates:list, state_sequence: np.array):
    N_A = number_of_substates[0]
    N_B = number_of_substates[1]

    seq = state_sequence.copy()

    mask_A = seq < N_A
    seq[mask_A] = -1 + seq[mask_A] / N_A
    seq[~mask_A] = (seq[~mask_A] - N_A + 1) / N_B
    
    return seq

# def function that hides the sequence of substates and identifies them with their "parent state" -1 or 1
def reduce_states(state_sequence):
    """
    Takes as input the array that contains the sequence of plotable states and identifies them with super states.

    """
    seq = state_sequence.copy()
    
    mask_A = seq < 0
    seq[mask_A] = -1
    seq[~mask_A] = 1

    return seq

def two_state_process_reduced(*args, **kwargs):
    time_sequence,state_sequence = generate_markov_jump_process(*args, **kwargs)
    number_of_substates = args[0]
    state_sequence = state_mapping(number_of_substates,state_sequence)
    state_sequence = reduce_states(state_sequence)
    return time_sequence,state_sequence

#the following function guarantees that the eigenvector that is chosen is a probability density, i.e. positive and normed to one
def norm_to_probability_density(x: np.array):
    if np.argmax(x) < 0:                   
        x = -x            
    x = 1/(np.sum(x)) * x
    return x

def build_generator_matrix(number_of_substates: list, transition_rates: list, refractory_periods: list):
    """
    Constructs the generator matrix (L) for the two-state renewal process
    with linear chains to model refractory periods.
    
    Args:
        number_of_substates (list): [N, M], the number of substates for state A and B.
        transition_rates (list): The final transition rates out of state A and B.
        refractory_periods (list): The desired mean refractory period for state A and B.

    Returns:
        array: The (N+M)x(N+M) generator matrix L.
    """
    #Define relevant parameters
    N = number_of_substates[0]
    M = number_of_substates[1]

    total_states = N + M

    transition_rates_substates_A = (N-1) / refractory_periods[0]
    transition_rates_substates_B = (M-1) / refractory_periods[1]

    transition_rate_final_A = transition_rates[0]
    transition_rate_final_B = transition_rates[1]

    #Initialize matrix
    L = np.zeros((total_states,total_states))

    for i in range(N-1):
        L[i,i] = (-1 * transition_rates_substates_A)
        L[i+1,i] = transition_rates_substates_A
    
    L[N-1,N-1] = (-1 * transition_rate_final_A)
    L[N,N-1] = transition_rate_final_A

    for i in range(N,N+M-1):
        L[i,i] = (-1 * transition_rates_substates_B)
        L[i+1,i] = transition_rates_substates_B

    L[N+M-1,N+M-1] = (-1 * transition_rate_final_B)
    L[0,N+M-1] = transition_rate_final_B

    return L


def calculate_and_order_eigenspectrum(L: np.array):
    """
    Calculates the eigenvalues of the generator matrix L.

    Args:  generator matrix L
    """
    #calculate lefthand eigenvalues
    eigenvalues_left_raw, eigenvectors_left = np.linalg.eig(L.T)     #since we want the lefthand eigenvectors, we take transpose the matrix here
    #Order eigenvalues by magnitude of real part. The resulting array starts with eigenvalues that have the most negative real part. 
    index_ordering = np.argsort(eigenvalues_left_raw.real)
    eigenvalues_left = eigenvalues_left_raw[index_ordering]
    #Since there are pairs of eigenvalues with same real part we have to make sure there is a unique ordering. Otherwise this is a source of error. We take the eigenvalues with positive imaginariy part first.
    for i in range(len(eigenvalues_left)-1):
        if eigenvalues_left[i].real == eigenvalues_left[i+1].real:
            if eigenvalues_left[i].imag > eigenvalues_left[i+1].imag:
                temp = index_ordering[i]
                index_ordering[i] = index_ordering[i+1]
                index_ordering[i+1] = temp
            else: continue
    eigenvalues_left = eigenvalues_left_raw[index_ordering]

    #Order eigenvectors according to the ordering of the eigenvalues
    ordered_eigenvectors_left = []
    for i in index_ordering:
        ordered_eigenvectors_left.append(eigenvectors_left[:,i])

    #Norm stationary left eigenvector to all ones
    for i in range(len(ordered_eigenvectors_left[-1])):
        ordered_eigenvectors_left[-1][i] = np.abs(1/(ordered_eigenvectors_left[-1][i]) * ordered_eigenvectors_left[-1][i])
    

    ################################################

    #calculate righthand eigenvalues
    eigenvalues_right_raw, eigenvectors_right = np.linalg.eig(L)
    
    idx_order = np.argsort(eigenvalues_right_raw.real)
    eigenvalues_right = eigenvalues_right_raw[idx_order]
    #Since there are pairs of eigenvalues with same real part we have to make sure there is a unique ordering. Otherwise this is a source of error. We take the eigenvalues with positive imaginariy part first.
    for i in range(len(eigenvalues_right)-1):
        if eigenvalues_right[i].real == eigenvalues_right[i+1].real:
            if eigenvalues_right[i].imag > eigenvalues_right[i+1].imag:
                temp = idx_order[i]
                idx_order[i] = idx_order[i+1]
                idx_order[i+1] = temp
            else: continue
    eigenvalues_right = eigenvalues_right_raw[idx_order]

    #Order eigenvectors according to the ordering of the eigenvalues
    ordered_eigenvectors_right = []
    for i in idx_order:
        ordered_eigenvectors_right.append(eigenvectors_right[:,i])

    #Norm stationary right eigenvector so that it is a probability distribution
    ordered_eigenvectors_right[-1] = norm_to_probability_density(ordered_eigenvectors_right[-1])

    #Norm the left eigenvectors Q_i to unit variance, as detailed in the paper eq. 9
    for i in range(len(ordered_eigenvectors_left)-1):
        variance = 0
        for j in range(len(ordered_eigenvectors_left[i])):
            variance += np.abs(ordered_eigenvectors_left[i][j])**2 * ordered_eigenvectors_right[-1][j]
        ordered_eigenvectors_left[i] = (1/np.sqrt(variance)) * ordered_eigenvectors_left[i]

    #Enforce biorthogonality 
    for i in range(len(ordered_eigenvectors_right)-1):
        dot_product = ordered_eigenvectors_right[i]@ordered_eigenvectors_left[i]
        ordered_eigenvectors_right[i] = (1/dot_product) * ordered_eigenvectors_right[i]

    return eigenvalues_left,ordered_eigenvectors_left,ordered_eigenvectors_right

def analyzing_and_plotting_eigenspectrum(ordered_eigenvalues, ordered_eigenvectors_left, ordered_eigenvectors_right):
    stationary_eigenvalue = ordered_eigenvalues[-1]

    #the eigenvalue of interest is the eigenvalue with the the largest non-zero real part
    lambda_1 = ordered_eigenvalues[-2]
    mu_1 = lambda_1.real
    omega_1 = lambda_1.imag

    #stationary left eigenvector
    Q_0 = ordered_eigenvectors_left[-1]
    #the following line guarantee that the eigenvalue that is chosen is a probability density, i.e. positive and normed to one

    #the left eigenvector Q_1 that corresponds to the lambda_1
    Q_1 = ordered_eigenvectors_left[-2]
    #Q_1 = norm_to_probability_density(Q_1)
    
    sum_entries_Q_0 = np.sum(Q_0)
    sum_entries_Q_1 = np.sum(Q_1)

    #stationary right eigenvector
    P_0 = ordered_eigenvectors_right[-1]
    #P_0 = norm_to_probability_density(P_0)

    #the right eigenvector P_1 that corresponds to the lambda_1
    P_1 = ordered_eigenvectors_right[-2]
    #P_1 = norm_to_probability_density(Q_1)
    
    sum_entries_P_0 = np.sum(P_0)
    sum_entries_P_1 = np.sum(P_1)


    print("-" * 30)
    print(f"Number of eigenvalues: {len(ordered_eigenvalues)}")
    print(f"Stationary Eigenvalue: {stationary_eigenvalue:.4f}")
    print(f"Oscillatory Eigenvalue (λ₁): {lambda_1:.4f}")
    print(f"Quality factor: {np.abs(omega_1/mu_1)}")
    print("-" * 30)
    print(f"Stationary lefthand Eigenvector: {Q_0}")
    print(" " * 30)
    print(f"Sum of elements of stationary lefthand eigenvector: {sum_entries_Q_0}")
    print(" " * 30)
    print(f"Oscillatory Eigenvector (Q₁): {Q_1}")
    print(" " * 30)
    print(f"Sum of elements of Q₁: {sum_entries_Q_1}")
    print("-" * 30)
    print(f"Stationary righthand Eigenvector: {P_0}")
    print(" " * 30)
    print(f"Sum of elements of stationary righthand eigenvector: {sum_entries_P_0}")
    print(" " * 30)
    print(f"Oscillatory Eigenvector (P₁): {P_1}")
    print(" " * 30)
    print(f"Sum of elements of P₁: {sum_entries_P_1}")


    #Plotting

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all eigenvalues
    ax.scatter(ordered_eigenvalues.real, ordered_eigenvalues.imag, c='teal', alpha=0.6, label='Other Eigenvalues')
    ax.scatter(ordered_eigenvalues.real, -ordered_eigenvalues.imag, c='teal', alpha=0.6) # Plot conjugates

    #Highlight eigenvalue with largest non-zero real part
    ax.scatter(lambda_1.real, lambda_1.imag, c='red', alpha=1, s=150, label=f'$\lambda_1$ = {mu_1:.2f} + {omega_1:.2f}i')
    ax.scatter(lambda_1.real, -lambda_1.imag, c='red', alpha=1, s=150) # Plot conjugates

    # Annotate the plot
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlim(-8.5, 0 + 0.5)
    ax.set_xlabel("Re(λ)", fontsize=14)
    ax.set_ylabel("Im(λ)", fontsize=14)
    ax.set_title("Eigenvalue Spectrum of the Generator Matrix", fontsize=16, weight='bold')
    ax.legend(fontsize=12)
    ax.grid(True)

    plt.show()

#calculate the conditional transition probability from the obtained eigenvetors
def transition_probability(eigenvalues: list, left_eigenvectors: list, right_eigenvectors: list ,initial_state: int,target_state: int,time: float):
        spectral_sum = 0
        for i in range(len(left_eigenvectors)-1):
            spectral_sum += np.exp(eigenvalues[i] * time) * right_eigenvectors[i][target_state] * left_eigenvectors[i][initial_state]
        return right_eigenvectors[-1][target_state] + spectral_sum

#define function that calculates power spectrum for given sequence of time and states by averaging over many realizations of the process
def power_spectrum(process_generator, realizations: int, *args, **kwargs):
    #process generator must return two arrays, one time sequence and one state sequence
    #call process generator once to initialize variables:
    time_sequence,state_sequence = process_generator(*args,**kwargs)
    if process_generator is generate_markov_jump_process:
        number_of_substates = args[0]                                           #### It is important now that number of substates is passed as first argument
        state_sequence = state_mapping(number_of_substates, state_sequence=state_sequence)
    #state_sequence = reduce_states(state_sequence)                              ####   if this line is uncommented the power spectrum of the superstates is calculated 
    number_sample_points = len(time_sequence)
    dt = time_sequence[1]-time_sequence[0]
    average = np.zeros(number_sample_points)
    #calculate first iteration with initialized variables
    ft_state_sequence = fft(state_sequence)*dt
    average += np.abs(ft_state_sequence)**2

    #calculate ensemble average over remaining realizations
    for i in range(1,realizations):
        time_sequence,state_sequence = process_generator(*args,**kwargs)
        if process_generator is generate_markov_jump_process:
            number_of_substates = args[0]  
            state_sequence = state_mapping(number_of_substates, state_sequence=state_sequence)
        #state_sequence = reduce_states(state_sequence)                          ####   if this line is uncommented the power spectrum of the superstates is calculated
        ft_state_sequence = fft(state_sequence)*dt
        if len(average) == len(ft_state_sequence):
            average += np.abs(ft_state_sequence)**2
        else: 
            print(f'Flag for realisation {i}')
            print(f'Mismatch: {len(average)} is not equal to {len(ft_state_sequence)}')

    #calculate final outputs
    frequency_sequence = fftfreq(number_sample_points,dt)
    power_spectrum_sequence = average/(realizations * (len(time_sequence)*dt))

    return frequency_sequence,power_spectrum_sequence

@njit
def phase_reduction(state_sequence: np.array, Q_1: np.array):
    
    Q_1_sequence = np.zeros(len(state_sequence), dtype=np.complex128)

    
    for i in range(len(state_sequence)):
        #reconstruct the state index from the trajectory value
        ### this assumes that the initial state is -1 and the indexing increases monotonically until state 1 is reached, i.e. -1 has index 0 and 1 has index 19
        x = state_sequence[i]
        if x < 0:
            advancement = np.abs(x - state_sequence[0])
            state_index = int(np.round(10*advancement,0))
        elif x > 0:
            advancement = x
            state_index = int(np.round(10*advancement,0) + 9)
        
        #assign calculated Q_1 value
        Q_1_sequence[i] = Q_1[state_index]
        
    complex_argument_Q_1 = np.angle(Q_1_sequence)

    return Q_1_sequence

def phase_reduced_process(time_sequence: list, state_sequence:list, Q_1: np.array):

    complex_argument_Q_1 = phase_reduction(state_sequence, Q_1)

    return time_sequence, complex_argument_Q_1

#the theoretically expected power spectrum of Q_1
def power_spectrum_theoretical_Q_1(eigenvalue: np.complex128, omega: float):
    mu_1 = eigenvalue.real
    omega_1 = eigenvalue.imag
    return (2*np.abs(mu_1)) / (mu_1**2 + (omega - omega_1)**2)

