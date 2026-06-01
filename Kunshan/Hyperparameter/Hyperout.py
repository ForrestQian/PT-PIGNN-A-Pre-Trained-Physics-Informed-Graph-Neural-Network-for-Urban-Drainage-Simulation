import pickle
import pandas as pd

TRIALS_PATH = ''  # Set locally before running
RESULTS_OUTPUT_PATH = ''  # Set locally before running

# Load Trials object
with open(TRIALS_PATH, 'rb') as f:
    trials = pickle.load(f)

# Init list to store each trial info
results = []

# Iterate over each trial
for trial in trials.trials:
    trial_info = {
        'tid': trial['tid'],  # trial ID
        'loss': trial['result']['loss'],  # loss value
        'params': trial['misc']['vals']  # hyperparameters
    }
    results.append(trial_info)

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)

# Display results
print(results_df)
results_df.to_excel(RESULTS_OUTPUT_PATH)
# Sort by loss to find best hyperparameter sets
best_results = results_df.nsmallest(10, 'loss')  # Top 10 trials with lowest loss
print("Best hyperparameter combinations:")
print(best_results)