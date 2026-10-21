from MSquantSim.downstreamanalysis import run_random_forest
import pandas as pd

# CRC data
crc_input = pd.read_csv("../data/crc/CRC_training_data.csv")

copula_sim_healthy = simulate_using_copula(
    crc_healthy, num_simulations=10, 
    samples_per_simulation=len(crc_healthy), # Set to size of input data to mirror variability
    output_dir=None, output_prefix='simulated',
    return_data=True, random_state=2
)

copula_sim_disease = simulate_using_copula(
    crc_disease, num_simulations=10, 
    samples_per_simulation=len(crc_disease), # Set to size of input data to mirror variability
    output_dir=None, output_prefix='simulated',
    return_data=True, random_state=100
)

run_random_forest(crc_input, "Condition", len(crc_input), True, n_runs=50)

crc_healthy = crc_input[crc_input['Condition'] == 'Healthy'].drop(columns=['Condition'])
crc_disease = crc_input[crc_input['Condition'] == 'CRC'].drop(columns=['Condition'])

# MEL data


# PDAC - DIA


# PDAC - MRM


