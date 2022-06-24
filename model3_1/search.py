import time 
from pathlib import Path 
from nni.experiment import Experiment
from termcolor import colored
import os

print(os.getcwd())
print(Path(__file__).parent)

#* Other hyperparams to tune, batch_size, sliding_window
search_space = {
    "learning_rate" : {"_type" : "choice", "_value" : [0.001, 0.005, 0.01]},
    "hidden_size" : {"_type" : "choice", "_value" : [16,32,64,128]},
    "optimizer" : {"_type" : "choice", "_value" : ["adam", "sgd", "adamax"]},
    "batch_size" : {"_type" : "choice", "_value" : [32,64,128,256]},
    "window_size" : {"_type" : "choice", "_value" : [24,72,168]},
    "model" : {"_type" : "choice", "_value" : ["gru", "lstm"]}
}

# Maximum number of trials
max_trials = 30

# ==============================================================================
# Search Configuration
# ==============================================================================
search = Experiment("local")

# Search name
search.config.experiment_name = "Model3 search" 

search.config.trial_concurrency = 2 # evaluates 2 hyperparams at a time (i think)
search.config.max_trial_number = max_trials # evaluates (30 - above) sets of hyperparams
search.config.search_space = search_space # hyperparm search space
search.config.trial_command = "python trial.py" # file name containing model training process
search.config.trial_code_directory = Path(__file__).parent # path to file

# Tuner settings
search.config.tuner.name = "Evolution" # Randomly initializes a population based on search space, for each generation it chooses better ones and does some mutations on them to get next generation
search.config.tuner.class_args["optimize_mode"] = "minimize" # tuner attempts to minimize the given metrics
search.config.tuner.class_args["population_size"] = 8 # "evolution" based metric, population_size > concurrency, The greater the size the better

search.start(8080)

executed_trials = 0 
while True:
    trials = search.export_data() # Returns exported information, #? Returns a list.  Its empty
    if executed_trials != len(trials):
        executed_trials = len(trials)
        print(f"\nTrials : {executed_trials} / {max_trials}") #- ?/30
    if search.get_status() == "DONE":
        best_trial = min(trials, key = lambda t: t.value)
        print(f"Best trial params: {best_trial.parameter}")
        input("Experiment is finished. Press any key to exit...")
        break 
    print(".", end = ""),
    time.sleep(10)