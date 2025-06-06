import numpy as np
import frustratometer
import pandas as pd  # Import pandas for data manipulation
import numba
from pathlib import Path

_AA = '-ACDEFGHIKLMNPQRSTVWY'

def index_to_sequence(seq_index):
    """Converts sequence index array back to sequence string."""
    return ''.join([_AA[index] for index in seq_index])

def sequence_to_index(sequence):
    """Converts sequence string to sequence index array."""
    return np.array([_AA.find(aa) for aa in sequence])

@numba.njit
def sequence_swap(seq_index, model_h, model_J, mask):
    seq_index = seq_index.copy()
    n=len(seq_index)
    res1 = np.random.randint(0,n)
    res2 = np.random.randint(0,n-1)
    res2 += (res2 >= res1)
    
    het_difference = 0
    energy_difference = compute_swap_energy(seq_index, model_h, model_J, mask, res1, res2)

    seq_index[res1], seq_index[res2] = seq_index[res2], seq_index[res1]

    return seq_index, het_difference, energy_difference

@numba.njit
def compute_swap_energy(seq_index, model_h, model_J, mask, pos1, pos2):
    aa2 , aa1 = seq_index[pos1],seq_index[pos2]
    
    #Compute fields
    energy_difference = 0
    energy_difference -= (model_h[pos1, aa1] - model_h[pos1, seq_index[pos1]])  # h correction aa1
    energy_difference -= (model_h[pos2, aa2] - model_h[pos2, seq_index[pos2]])  # h correction aa2
    
    #Compute couplings
    j_correction = 0.0
    for pos in range(len(seq_index)):
        aa = seq_index[pos]
        # Corrections for interactions with pos1 and pos2
        j_correction += model_J[pos, pos1, aa, seq_index[pos1]] * mask[pos, pos1]
        j_correction -= model_J[pos, pos1, aa, aa1] * mask[pos, pos1]
        j_correction += model_J[pos, pos2, aa, seq_index[pos2]] * mask[pos, pos2]
        j_correction -= model_J[pos, pos2, aa, aa2] * mask[pos, pos2]

    # J correction, interaction with self aminoacids
    j_correction -= model_J[pos1, pos2, seq_index[pos1], seq_index[pos2]] * mask[pos1, pos2]  # Taken two times
    j_correction += model_J[pos1, pos2, aa1, seq_index[pos2]] * mask[pos1, pos2]  # Added mistakenly
    j_correction += model_J[pos1, pos2, seq_index[pos1], aa2] * mask[pos1, pos2]  # Added mistakenly
    j_correction -= model_J[pos1, pos2, aa1, aa2] * mask[pos1, pos2]  # Correct combination
    energy_difference += j_correction
    return energy_difference


@numba.njit
def sequence_mutation(seq_index, model_h, model_J, mask,valid_indices=np.arange(len(_AA))):
    seq_index = seq_index.copy()
    r = np.random.randint(0, len(valid_indices)*len(seq_index)) # Select a random index
    res = r // len(valid_indices)
    aa_new = valid_indices[r % len(valid_indices)] 

    aa_old_count = np.sum(seq_index == seq_index[res])
    aa_new_count = np.sum(seq_index == aa_new)
    
    het_difference = np.log(aa_old_count / (aa_new_count+1)) 
    energy_difference = compute_mutation_energy(seq_index, model_h, model_J, mask, res, aa_new)

    seq_index[res] = aa_new
    
    return seq_index, het_difference, energy_difference

@numba.njit
def compute_mutation_energy(seq_index: np.ndarray, model_h: np.ndarray, model_J: np.ndarray, mask: np.ndarray, pos: int, aa_new: int) -> float:
    aa_old=seq_index[pos]
    energy_difference = -model_h[pos,aa_new] + model_h[pos,aa_old]

    energy_difference = -model_h[pos, aa_new] + model_h[pos, aa_old]

    # Initialize j_correction to 0
    j_correction = 0.0

    # Manually iterate over the sequence indices
    for idx in range(len(seq_index)):
        aa_idx = seq_index[idx]  # The amino acid at the current position
        # Accumulate corrections for positions other than the mutated one
        j_correction += model_J[idx, pos, aa_idx, aa_old] * mask[idx, pos]
        j_correction -= model_J[idx, pos, aa_idx, aa_new] * mask[idx, pos]

    # For self-interaction, subtract the old interaction and add the new one
    j_correction -= model_J[pos, pos, aa_old, aa_old] * mask[pos, pos]
    j_correction += model_J[pos, pos, aa_new, aa_new] * mask[pos, pos]

    energy_difference += j_correction

    return energy_difference

@numba.njit
def model_energy(seq_index: np.array,
                  model_h: np.ndarray, model_J: np.ndarray,
                  mask: np.array) -> float:
    seq_len = len(seq_index)
    energy_h = 0.0
    energy_J = 0.0

    for i in range(seq_len):
        energy_h -= model_h[i, seq_index[i]]
    
    for i in range(seq_len):
        for j in range(seq_len):
            aa_i = seq_index[i]
            aa_j = seq_index[j]
            energy_J -= model_J[i, j, aa_i, aa_j] * mask[i, j]
    
    total_energy = energy_h + energy_J / 2
    return total_energy

def heterogeneity(seq_index):
    N = len(seq_index)
    _, counts = np.unique(seq_index, return_counts=True)
    denominator = np.prod(np.array([np.math.factorial(count) for count in counts]))
    het = np.math.factorial(N) / denominator
    return np.log(het)

log_factorial_table=np.log(np.array([np.math.factorial(i) for i in range(40)],dtype=np.float64))

@numba.njit
def stirling_log(n):
    if n < 40:
        return log_factorial_table[n]
    else:
        return n * np.log(n / np.e) + 0.5 * np.log(2 * np.pi * n) + 1.0 / (12 * n)

@numba.njit
def heterogeneity_approximation(seq_index):
    """
    Uses Stirling's approximation to calculate the heterogeneity of a sequence
    """
    N = len(seq_index)
    counts = np.zeros(21, dtype=np.int32)
    
    for val in seq_index:
        counts[val] += 1
        
    log_n_factorial = stirling_log(N)
    log_denominator = sum([stirling_log(count) for count in counts])
    het = log_n_factorial - log_denominator
    return het

@numba.njit
def montecarlo_steps(temperature, model_h, model_J, mask, seq_index, Ep=100, n_steps = 1000, kb = 0.008314,valid_indices=np.arange(len(_AA))) -> np.array:
    for _ in range(n_steps):
        new_sequence, het_difference, energy_difference = sequence_swap(seq_index, model_h, model_J, mask) if np.random.random() > 0.5 else sequence_mutation(seq_index, model_h, model_J, mask,valid_indices)
        exponent=(-energy_difference + Ep * het_difference) / (kb * temperature + 1E-10)
        acceptance_probability = np.exp(min(0, exponent)) 
        if np.random.random() < acceptance_probability:
            seq_index = new_sequence
    return seq_index

@numba.njit
def replica_exchanges(energies, temperatures, kb=0.008314, exchange_id=0):
    """
    Determine pairs of configurations between replicas for exchange.
    
    Returns a list of tuples with the indices of replicas to be exchanged.
    """
    n_replicas = len(temperatures)
    start_index = exchange_id % 2
    order = np.arange(len(temperatures), dtype=np.int64)
    
    for i in np.arange(start_index, n_replicas - 1, 2):
        energy1, energy2 = energies[i], energies[i + 1]
        temp1, temp2 = temperatures[i], temperatures[i + 1]
        delta = (1/temp2 - 1/temp1) * (energy2 - energy1)
            
        exponent = delta / kb # Sign is correct, as we want to swap when the system with higher temperature has lower energy
        prob = np.exp(min(0., exponent)) 

        #if 1 <= prob:
        acceptance_probability = np.exp(min(0, exponent))
        if np.random.random() < acceptance_probability:
            order[i]=i+1
            order[i+1]=i
    return order

@numba.njit(parallel=True)
def parallel_montecarlo_step(model_h, model_J, mask, seq_indices, temperatures, n_steps_per_cycle, Ep,valid_indices=np.arange(len(_AA))):
    n_replicas = len(temperatures)
    energies = np.zeros(n_replicas)
    heterogeneities = np.zeros(n_replicas)
    total_energies = np.zeros(n_replicas)
    for i in numba.prange(n_replicas):
        temp_seq_index = seq_indices[i]
        seq_indices[i] = montecarlo_steps(temperatures[i], model_h, model_J, mask, seq_index=temp_seq_index, Ep=Ep, n_steps=n_steps_per_cycle,valid_indices=valid_indices)
        energy = model_energy(seq_indices[i], model_h, model_J, mask)
        het = heterogeneity_approximation(seq_indices[i])
        # Compute energy for the new sequence
        total_energies[i] = energy - Ep * het # Placeholder for actual energy calculation
        energies[i] = energy
        heterogeneities[i] = het

    
    return seq_indices, energies, heterogeneities, total_energies

@numba.njit
def parallel_tempering_numba(model_h, model_J, mask, seq_indices, temperatures, n_steps, n_steps_per_cycle, Ep,valid_indices=np.arange(len(_AA))):
    for s in range(n_steps//n_steps_per_cycle):
        seq_indices, energy, het, total_energies = parallel_montecarlo_step(model_h, model_J, mask, seq_indices, temperatures, n_steps_per_cycle, Ep,valid_indices=valid_indices)

        # Yield data every 10 exchanges
        if s % 10 == 9:
            yield s, seq_indices, energy, het, total_energies

        # Perform replica exchanges
        order = replica_exchanges(total_energies, temperatures, exchange_id=s)
        seq_indices = seq_indices[order]
        


def parallel_tempering(model_h, model_J, mask, seq_indices, temperatures, n_steps, n_steps_per_cycle, Ep, filename="parallel_tempering_resultsv3.csv",valid_indices=np.arange(len(_AA))):
    columns=['Step', 'Temperature', 'Sequence', 'Energy', 'Heterogeneity', 'Total Energy']
    df_headers = pd.DataFrame(columns=columns)
    df_headers.to_csv(filename, index=False)
    print(*columns, sep='\t')

    # Run the simulation and append data periodically
    for s, updated_seq_indices, energy, het, total_energy in parallel_tempering_numba(model_h, model_J, mask, seq_indices, temperatures, n_steps, n_steps_per_cycle, Ep,valid_indices=valid_indices):
        # Prepare data for this chunk
        data_chunk = []
        for i, temp in enumerate(temperatures):
            sequence_str = index_to_sequence(updated_seq_indices[i])  # Convert sequence index back to string
            #total_energy = energy[i] - Ep * het[i]
            data_chunk.append({'Step': (s+1) * n_steps_per_cycle, 'Temperature': temp, 'Sequence': sequence_str, 'Energy': energy[i], 'Heterogeneity': het[i], 'Total Energy': total_energy[i]})
        
        # Convert the chunk to a DataFrame and append it to the CSV
        df_chunk = pd.DataFrame(data_chunk)
        print(*df_chunk.iloc[-1].values, sep='\t')
        df_chunk.to_csv(filename, mode='a', header=False, index=False)


def annealing(temp_max=500, temp_min=0, n_steps=1E8, Ep=10,valid_indices=np.arange(len(_AA))):
    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    seq_index = sequence_to_index("SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLPELASALGVSVDWLLNGT")
    
    simulation_data = []
    n_steps_per_cycle=n_steps//(temp_max-temp_min)
    for temp in range(temp_max, temp_min, -1):
        seq_index= montecarlo_steps(temp, model.potts_model['h'], model.potts_model['J'], model.mask, seq_index, Ep=Ep, n_steps=n_steps_per_cycle,valid_indices=valid_indices)
        energy = model_energy(seq_index, model.potts_model['h'],model.potts_model['J'], model.mask)
        het = heterogeneity_approximation(seq_index)
        simulation_data.append({'Temperature': temp, 'Sequence': index_to_sequence(seq_index), 'Energy': energy, 'Heterogeneity': het, 'Total Energy': energy - Ep * het})
        print(temp, index_to_sequence(seq_index), energy - Ep * het, energy, het)
    simulation_df = pd.DataFrame(simulation_data)
    simulation_df.to_csv("mcso_simulation_results.csv", index=False)

def benchmark_montecarlo_steps(n_repeats=100, n_steps=20000,valid_indices=np.arange(len(_AA))):
    import time
    # Initialize the model for 1r69
    native_pdb = "tests/data/1r69.pdb"  # Ensure this path is correct
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2)
    
    seq_len = len(model.sequence)
    times = []

    #Adds one step for numba compilation time
    montecarlo_steps(temperature=500, model_h=model.potts_model['h'], model_J=model.potts_model['J'], mask=model.mask, seq_index=sequence_to_index(model.sequence), Ep=100, n_steps=1,valid_indices=valid_indices)

    for _ in range(n_repeats):  # Run benchmark 10 times
        # Generate a new random sequence for each run
        seq_index = np.random.randint(1, 21, size=seq_len)
        start_time = time.time()
        
        montecarlo_steps(temperature=500, model_h=model.potts_model['h'], model_J=model.potts_model['J'], mask=model.mask, seq_index=seq_index, Ep=100, n_steps=n_steps,valid_indices=valid_indices)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    average_time_per_step_s = sum(times) / len(times) / n_steps
    average_time_per_step_us = average_time_per_step_s * 1000000
    
    steps_per_hour = 3600 / average_time_per_step_s
    minutes_needed = 1E10 / steps_per_hour / 8 * 60  # 8 processes in parallel

    print(f"Time needed to explore 10^10 sequences with 8 process in parallel: {minutes_needed:.2e} minutes")
    print(f"Number of sequences explored per hour: {steps_per_hour:.2e}")
    print(f"Average execution time per step: {average_time_per_step_us:.5f} microseconds")

if __name__ == '__main__':
    excluded = '-CP'
    # lists are mutable so must be converted to tuple for parallelization
    # when declared in global scope
    valid_indices = tuple([i for i,aa in enumerate(_AA) if aa not in excluded]) 
    #len_valid_indices = len(valid_indices)

    benchmark_montecarlo_steps()
    #annealing(n_steps=1E6,valid_indices=valid_indices)
    import warnings
    import numpy as np

    # Convert overflow warnings to exceptions
    warnings.filterwarnings('error', 'overflow encountered in power', category=RuntimeWarning)

    native_pdb = "tests/data/1r69.pdb"
    structure = frustratometer.Structure.full_pdb(native_pdb, "A")
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=2) # intial local sequence term
    #model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=5) # peter's recommendation
    #model = frustratometer.AWSEM(structure, distance_cutoff_contact=10, min_sequence_separation_contact=10) # like the usual awsem force field


    # make sure w
    seq_index = sequence_to_index("SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLPELASALGVSVDWLLNGT")

    temperatures=np.logspace(0,6,49)
    seq_indices=np.random.randint(0, len(valid_indices), size=(len(temperatures),len(model.sequence)))
    print(len(seq_indices))
    for i,aa in enumerate(_AA):
        if aa in excluded:
            seq_indices[seq_indices>=i] += 1
    parallel_tempering(model.potts_model['h'], model.potts_model['J'], model.mask, seq_indices, temperatures, n_steps=int(1E10), n_steps_per_cycle=int(1E4), Ep=30,valid_indices=valid_indices,filename="ivan_test.csv")
