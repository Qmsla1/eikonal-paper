# -*- coding: utf-8 -*-
"""
runner.py

This is the runner file that creates a fresh experiment and runs the training and evaluation
for each dimension. It uses the experiment module and combines final results from all dimensions.

All printed output is duplicated into a single log file saved in UTF-8.
In the terminal, the output appears in red.
However, to avoid embedding ANSI escape codes in the log file, the log file will contain plain text.
This mimics the behavior of shell redirection ">> $(date +%Y%m%d_%H%M%S)_output.log 2>&1" without color codes in the file.
"""

import os
import sys
import traceback

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch import nn
from torch.optim import LBFGS , Adam
from experiment import ExperimentManager, PINN, train_pinn, evaluate_model

# --- Simple Tee implementation ---
class Tee:
    def __init__(self, terminal_stream, file_stream, color=True):
        self.terminal = terminal_stream
        self.file = file_stream
        self.color = color
        # ANSI escape sequences for red text for terminal only.
        self.RED = "\033[31m" if color else ""
        self.RESET = "\033[0m" if color else ""

    def write(self, message):
        # Write colored message to terminal.
        self.terminal.write(f"{self.RED}{message}{self.RESET}")
        self.terminal.flush()
        # Write plain message to file.
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()


# --- Setup Experiment Manager and Logging ---
# Base configuration.
config = {
    'dimensions_to_test': list(range(3, 16)),  # For example, testing dimension 10.
    'n_neurons': 200,
    'output_dim': 1,
    'num_hidden_layers': 5,
    'num_epochs': 1000,
    'pretrain_epochs': 200,
    'manifold_type': 'circle',
    'n_train_int': 100000,
    'n_train_b': 10000,
    'n_train_b_val': 100,
    'optimizer_config': {
        'lr': 1,
        'max_iter': 20,
        'max_eval': 25,
        'tolerance_grad': 1e-7,
        'tolerance_change': 1e-9,
        'history_size': 50,
        'line_search_fn': 'strong_wolfe'
    },
    'solver': {
        'type': 'aug_lagrangian',
        'mu_initial': 0.1,
        'mu_max': 1e6,
        'mu_growth': 1.0,
        'dual_step': 0.01,
        'dual_clip': [0.0, 1e6],
        'update_every': 1
    },
    'objective': {
        'type': 'exp_neg_mean',
        'temperature': 0.1,
        'quantile': 0.1
    },
    'training': {
        'grad_clip_norm': None
    }
}

# Initialize experiment manager.
exp_manager = ExperimentManager(config['manifold_type'], config)
experiment_dir = exp_manager.exp_dir
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

# Create a log filename with timestamp.
log_filename = os.path.join(experiment_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_output.log")
# Open the log file for writing with UTF-8 encoding.
log_file = open(log_filename, 'w', encoding='utf-8')

# Override both sys.stdout and sys.stderr to duplicate output.
# Terminal output will appear in red; log file output will be plain.
sys.stdout = Tee(sys.__stdout__, log_file, color=True)
sys.stderr = Tee(sys.__stderr__, log_file, color=True)

# --- Print initial info ---
print("Experiment directory:", experiment_dir)

# --- Set GPU configuration ---
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"  # Exclude GPU 0

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    dev = torch.device("cuda")
    dev0 = torch.device("cuda:0")
    print(f"Using {num_gpus} GPUs")
else:
    num_gpus = 0
    dev = torch.device("cpu")
    dev0 = dev
    print("Using CPU")

# --- Training and Evaluation ---
all_results = {}

dims_errors = []

for dim in config['dimensions_to_test']:
    print("\n" + "="*50)
    print(f"Training model for dimension {dim}")
    print("="*50)
    try:
        # Initialize model.
        model = PINN(dim, config['n_neurons'], config['output_dim'], config['num_hidden_layers']).to(dev)
        if num_gpus > 1:
            model = nn.DataParallel(model)
        # Initialize optimizers: Adam pretraining followed by LBFGS fine-tuning
        optimizer = LBFGS(model.parameters(), **config['optimizer_config'])
        pretrain_optimizer = Adam(model.parameters(), lr=1e-3)
        # Import manifold instance.
        from manifolds.manifold_factory import ManifoldFactory
        manifold = ManifoldFactory.get_instance_direct(dev, config['manifold_type'], output_dim=dim)
        # Prepare training data.
        from experiment import sample_x_train_b, sample_x_train_int
        x_train_b = sample_x_train_b(n_points=config['n_train_b'], manifold=manifold).to(dev0)
        x_train_int = sample_x_train_int(n_points=config['n_train_int'], manifold=manifold, r=0, r_max=20, dev=dev0).to(dev0)
        # Train the model.
        model, metrics_collector = train_pinn(
            model,
            optimizer,
            x_train_int,
            x_train_b,
            config['num_epochs'],
            exp_manager,
            dim,
            dev,
            pretrain_optimizer=pretrain_optimizer,
            pretrain_epochs=config['pretrain_epochs']
        )
        # Save best model.
        model_path = exp_manager.get_path('models', f'best_model_dim_{dim}.pth')
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
        print(f"Saved model for dimension {dim}")
    except Exception as e:
        dims_errors.append(f"exception_train_dim_{dim}")
        print(f"Error during training for dimension {dim}: {str(e)}")
        print(traceback.format_exc())
        raise e
    try:
        # Evaluate the model.
        mean_error, std_error = evaluate_model(model, dim, exp_manager, dev)
        all_results[dim] = {'mean_error': mean_error, 'std_error': std_error}
    except Exception as e:
        dims_errors.append(f"exception_evaluation_dim_{dim}")
        print(f"Error during evaluation for dimension {dim}: {str(e)}")
        print(traceback.format_exc())
        continue

# --- Create Combined Plot ---
try:
    plt.figure(figsize=(12, 8))
    test_range_vec = np.arange(1, 11)
    for dim in all_results:
        plt.plot(test_range_vec, all_results[dim]['mean_error'], label=f'Dim {dim}', marker='o')
    plt.xlabel('Test Range')
    plt.ylabel('Mean L1 Error')
    plt.title('Mean L1 Error Comparison Across Dimensions')
    plt.grid(True)
    plt.legend()
    combined_plot_path = exp_manager.get_path('plots', 'combined_error_plot.png')
    plt.savefig(combined_plot_path)
    plt.close()
    np.savez(exp_manager.get_path('results', 'combined_results.npz'),
             all_results=all_results,
             test_range=test_range_vec)
    print("Combined results saved successfully.")
except Exception as e:
    print(f"Error creating combined plot: {str(e)}")

if len(dims_errors) ==0:
    print("\nExperiment completed successfully!")
else:
    print("\nExperiment completed with errors in the following dimensions:")
    print(dims_errors)

print(f"Results saved in: {experiment_dir}")