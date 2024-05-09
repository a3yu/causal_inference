# Causal Inference

Python module that runs causal inference experiments, implementing procedures as outlined in the paper [Staggered Rollout Designs Enable Causal Inference Under Interference Without Network Knowledge
](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3103b25853719847502559bf67eb4037-Abstract-Conference.html).

## Module Structure
```bash
.
├── experiment
│   ├── estimators.py
│   ├── rct.py # randomised control trials
├── model
│   ├── graph.py # OOP graph implementations
│   ├── graphs.py # basic graph implementations
│   ├── pom.py # potential outcome models
├── plots # storage of experiment plots
├── runners # experiment run files
│   ├── experiment.py # univariate experiment
│   ├── main_experiment_two-stage.py # TODO
│   ├── modular_experiment.py # TODO
│   ├── parallel_experiment.py # TODO
└── tests # unit tests
```


## Usage instructions
### First time setting up the environment
To get started, first clone and install the necessary requirements using the following command:
```
git clone git@github.com:a3yu/causal_inference.git
```
Make sure you are in a Python >=3.8 environment.

(Optional): create a Python virtual environment to isolate your dependencies for this repository, then activate it:
```
python3 -m venv venv
source venv/bin/activate
```
Navigate to your local repository and install dependencies:
```
python3 -m pip install -r requirements.txt
```
### Running an experiment
To run a univariate experiment, simply run `python3 runners/experiment.py` and edit the run_experiment(), sample usage, and plotting fields accordingly.

### Overview of Experiment
One iteration of an experiment:
1. first, generate a graph using a specified graph model (SBM, ER, etc).
2. Use a specified potential outcomes model (e.g. polynomial POM) to compute the total treatment effect and a function to produce outcomes 
on the graph based on a treatment tensor.
3. Compute a treatment tensor based on a randomised control trial (e.g. staggered Bernoulli).
4. Compute a TTE estimate using a specified estimator (e.g. polynomial estimate) using the above treatment and outcome tensors.
5. Repeat this above process over varying time steps (t) and over varying number of repetitions (r), and compute the bias and variance
of the estimate vs the true TTE. 
