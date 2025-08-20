"""
This module contains the experiment infrastructure:
 - ExperimentManager: Manages experiment directories and configuration saving.
 - MetricsCollector: Saves training/test metrics and plots loss ratio evolution.
 - ModelCheckpoint: Keeps track of best models (loss / accuracy) and saves them.
 - EarlyStopping: A standard early stopping mechanism.
 - EnhancedEarlyStopping: A custom early stopping mechanism based on stabilization of a ratio.
 - PINN model and helper functions for data sampling, training and evaluation.
 - Additional utility functions.
"""

from datetime import datetime
import torch
from torch import nn
import torch.optim
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import copy
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union



# Utility function to unwrap a state_dict from DataParallel.
def unwrap_state_dict(state_dict):
    # Remove 'module.' prefix if it exists.
    if any(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


class EnhancedEarlyStopping:
   def __init__(self, warmup=30, patience=15, monitor_window=8, threshold=0.01, max_counter=30):
       """
       Args:
           warmup: Number of epochs to wait before starting monitoring
           patience: Number of epochs to wait after significant increase is detected
           monitor_window: Window size for calculating moving averages
           threshold: Threshold for significant increase (e.g. 0.1 = 10% increase)
       """
       self.warmup = warmup
       self.patience = patience
       self.monitor_window = monitor_window
       self.threshold = threshold  # Threshold for significant increase detection
       self.max_counter = max_counter
       self.best_metric = float('inf')
       self.best_state = None
       self.best_epoch = None
       self.counter = 0
       self.metric_history = []
       self.started_monitoring = False
       self._remove_todo_force_stop = False
       
   def calculate_significant_increase(self):
       """
       Check if there is a significant increase in the metric by comparing
       current moving average to previous window average.
       Returns True if increase is above threshold.
       """
       if len(self.metric_history) < self.monitor_window:
           return False
       
       recent_metrics = self.metric_history[-self.monitor_window:]
       
       # Calculate moving averages
       moving_avg = np.mean(recent_metrics)
       baseline_avg = np.mean(self.metric_history[:-self.monitor_window][-self.monitor_window:])
       
       # Calculate percent change
       percent_change = (moving_avg - baseline_avg) / baseline_avg
       
       # Check if increase is significant
       return percent_change > self.threshold
       
   def update(self, epoch, loss_values, state_dict):
       """
       Update early stopping state and check if training should stop.
       
       Args:
           epoch: Current training epoch
           loss_values: Dictionary containing loss components
           state_dict: Current model state
           
       Returns:
           stop_training (bool): Whether to stop training
           best_state (dict): Best model state if stopping, None otherwise
       """
       if epoch < self.warmup or self._remove_todo_force_stop :
           return False, None
           
       current_metric = loss_values['lagrangian_term'] / loss_values['true_loss']
       self.metric_history.append(current_metric)
       
       if not self.started_monitoring and epoch >= self.warmup:
           self.started_monitoring = True
           print(f"Started monitoring at epoch {epoch}")
           
       if self.started_monitoring:
           # Update best metric
           if current_metric < self.best_metric:
               self.best_metric = current_metric
               self.best_state = copy.deepcopy(state_dict)
               self.best_epoch = epoch
               self.counter = 0
           else:
               self.counter += 1
               
           # Check for significant increase
           if len(self.metric_history) >= 2 * self.monitor_window:
               has_significant_increase = self.calculate_significant_increase()
               
               if has_significant_increase and self.counter >= self.patience or self.counter >= self.patience and self.counter >= self.max_counter:
                   print(f"Early stopping triggered due to significant increase={has_significant_increase} Best metric: {self.best_metric:.6f} at epoch {self.best_epoch}")
                   # todo remove for debuging only prints
                   # return True, self.best_state
                   self._remove_todo_force_stop = True
                   return False, None
                   
       return False, None

#############################################
# Experiment Infrastructure
#############################################
class ExperimentManager:
    def __init__(self, manifold_type, base_config):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.manifold_type = manifold_type
        self.base_config = base_config

        # Main experiment directory.
        self.exp_dir = os.path.join('experiments', f'exp_{self.manifold_type}_{self.timestamp}')

        # Subdirectories.
        self.subdirs = {
            'metrics': os.path.join(self.exp_dir, 'metrics'),
            'models': os.path.join(self.exp_dir, 'models'),
            'plots': os.path.join(self.exp_dir, 'plots'),
            'results': os.path.join(self.exp_dir, 'results'),
            'checkpoints': os.path.join(self.exp_dir, 'checkpoints')
        }
        for dir_path in self.subdirs.values():
            os.makedirs(dir_path, exist_ok=True)
        self.save_config()

    def save_config(self):
        config_path = os.path.join(self.exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.base_config, f, indent=4)

    def get_path(self, subdir, filename):
        return os.path.join(self.subdirs[subdir], filename)


class MetricsCollector:
    def __init__(self, exp_manager, dimension, mode):
        self.exp_manager = exp_manager
        self.dimension = dimension
        self.mode = mode  # 'train' or 'test'
        self.metrics = []
        self.csv_file = self.exp_manager.get_path('metrics', f'{exp_manager.manifold_type}_dim_{dimension}_{mode}.csv')

    def add_metric(self, epoch, **kwargs):
        metrics_dict = {'epoch': epoch}
        metrics_dict.update(kwargs)
        self.metrics.append(metrics_dict)
        df = pd.DataFrame([metrics_dict])
        if os.path.exists(self.csv_file):
            df.to_csv(self.csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.csv_file, mode='w', header=True, index=False)

    def plot_metrics(self):
        if not self.metrics:   # nothing logged
            print(f"No metrics collected for {self.mode} in dim {self.dimension}, skipping plot.")
        return

        df = pd.DataFrame(self.metrics)
        
        plt.figure(figsize=(12, 6))
        
        # Plot training accuracy
        ax1 = plt.gca()
        line1 = ax1.plot(df['epoch'], df['accuracy_error'], 'b-', label='Training Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training Accuracy', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Plot loss_int/loss_dmax ratio on secondary y-axis
        ax2 = ax1.twinx()
        ratio = df['loss_int'] / df['loss_dmax']
        line2 = ax2.plot(df['epoch'], ratio, 'r-', label='loss_int/loss_dmax ratio')
        ax2.set_ylabel('Loss Ratio', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Combine lines for legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        plt.title(f'Training Metrics - {self.exp_manager.manifold_type} Dim {self.dimension}')
        plt.grid(True)
        
        plot_path = self.exp_manager.get_path('plots',
                                           f'training_metrics_{self.exp_manager.manifold_type}_dim_{self.dimension}.png')
        plt.savefig(plot_path)
        plt.close()

    def save_all_metrics(self):
        if not self.metrics:   # nothing logged
            print(f"No metrics to save for {self.mode} in dim {self.dimension}.")
        return
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.csv_file, mode='w', header=True, index=False)


class ModelCheckpoint:
    def __init__(self, exp_manager, data_dim, max_keep=3):
        self.data_dim = data_dim
        self.max_keep = max_keep
        self.best_loss_models = []  # (loss, epoch, state_dict)
        self.best_accuracy_models = []  # (accuracy, epoch, state_dict)
        self.exp_manager = exp_manager
        self.checkpoint_dir = os.path.join(self.exp_manager.subdirs['checkpoints'], f'dim_{self.data_dim}')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def update(self, model, epoch, loss, accuracy):
        if isinstance(model, nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        self.best_loss_models.append((loss, epoch, copy.deepcopy(state_dict)))
        self.best_loss_models.sort(key=lambda x: x[0])
        self.best_loss_models = self.best_loss_models[:self.max_keep]

        self.best_accuracy_models.append((accuracy, epoch, copy.deepcopy(state_dict)))
        self.best_accuracy_models.sort(key=lambda x: x[0])
        self.best_accuracy_models = self.best_accuracy_models[:self.max_keep]

    def save_best_models(self, base_filename):
        for i, (loss, epoch, state_dict) in enumerate(self.best_loss_models):
            filename = os.path.join(self.checkpoint_dir, f"{base_filename}_loss_{i + 1}_epoch_{epoch}.pth")
            torch.save({'epoch': epoch, 'loss': loss, 'state_dict': state_dict}, filename)
        for i, (acc, epoch, state_dict) in enumerate(self.best_accuracy_models):
            filename = os.path.join(self.checkpoint_dir, f"{base_filename}_acc_{i + 1}_epoch_{epoch}.pth")
            torch.save({'epoch': epoch, 'accuracy': acc, 'state_dict': state_dict}, filename)


#############################################
# PINN Model and Helper Functions
#############################################
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)


def distance_function(x):
    norm_circle_plane = torch.norm(x[:, :2], dim=1)
    if x.shape[1] == 2:
        return torch.abs(norm_circle_plane - 1)
    else:
        norm_other_dims = torch.norm(x[:, 2:], dim=1)
        return torch.sqrt((norm_circle_plane - 1) ** 2 + norm_other_dims ** 2)


def sample_x_train_b(n_points, manifold):
    return manifold.sample_n_points(n_points)


def sample_x_train_int(n_points, manifold, r=1.0, r_max=2.0, dev=torch.device("cpu")):
    data = 1 - 2 * torch.rand([n_points, manifold.output_dim], device=dev)
    data = data / data.norm(dim=1, keepdim=True)
    if r_max != 0:
        random_radius = r + (r_max - r) * torch.rand([n_points, 1], device=dev)
    else:
        random_radius = r * torch.rand([n_points, 1], device=dev)
    data = random_radius * data
    return data





@dataclass
class LossTerm:
    """
    A class representing a term in the loss function.

    Attributes:
        name: Unique name of the loss term
        compute_fn: Function to compute the loss value
        is_constraint: Whether this term is a constraint (True) or objective (False)
        target_value: Target value for the constraint (usually 0)
        tolerance: Tolerance for constraint satisfaction
        use_lagrange_multiplier: Whether to use a Lagrange multiplier for this term
        use_penalty: Whether to use penalty term for this term
        weight: Fixed weight for this term (used when not using Lagrange multiplier)
        lagrange_multiplier: Lagrange multiplier tensor (initialized if use_lagrange_multiplier=True)
    """
    name: str
    compute_fn: Callable
    is_constraint: bool = False
    target_value: float = 0.0
    tolerance: float = 1e-4
    use_lagrange_multiplier: bool = False
    use_penalty: bool = False
    weight: float = 1.0
    lagrange_multiplier: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.use_lagrange_multiplier and self.lagrange_multiplier is None:
            self.lagrange_multiplier = torch.tensor(1.0, requires_grad=True)

    def compute(self, *args, **kwargs) -> torch.Tensor:
        """Compute the loss value"""
        return self.compute_fn(*args, **kwargs)

    def get_weight(self) -> Union[float, torch.Tensor]:
        """Get the weight or Lagrange multiplier for this term"""
        if self.use_lagrange_multiplier:
            return self.lagrange_multiplier
        return self.weight

    def get_violation(self, value: torch.Tensor) -> float:
        """Compute constraint violation"""
        if not self.is_constraint:
            return 0.0
        return max(0.0, value - self.tolerance)


class ModularAugmentedLagrangian:
    """
    A modular implementation of the augmented Lagrangian method.

    This class manages multiple loss terms, computes the augmented Lagrangian,
    and updates the Lagrange multipliers and penalty parameter.
    """

    def __init__(self, model: nn.Module, loss_terms: List[LossTerm],
                 mu_initial: float = 0.1, mu_max: float = 1e6):
        """
        Initialize the augmented Lagrangian manager.

        Args:
            model: The model being trained
            loss_terms: List of loss terms
            mu_initial: Initial penalty parameter
            mu_max: Maximum penalty parameter
        """
        self.model = model
        self.loss_terms = {term.name: term for term in loss_terms}
        self.mu = mu_initial
        self.mu_max = mu_max

    def add_loss_term(self, term: LossTerm) -> None:
        """Add a new loss term"""
        self.loss_terms[term.name] = term

    def compute_loss(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the augmented Lagrangian loss.

        Returns:
            total_loss: The total loss value
            loss_values: Dictionary of all computed loss values
        """
        # Compute individual loss terms
        loss_values = {}
        for name, term in self.loss_terms.items():
            loss_values[name] = term.compute(*args, **kwargs)

        # Compute true loss (sum of all terms without multipliers)
        true_loss = sum(loss_values.values())
        loss_values['true_loss'] = true_loss.item()

        # Compute Lagrangian term
        lagrangian_term = sum(
            term.get_weight() * loss_values[name]
            for name, term in self.loss_terms.items()
            if term.use_lagrange_multiplier or not term.is_constraint
        )
        loss_values['lagrangian_term'] = lagrangian_term.item()

        # Compute augmented term
        augmented_term = (self.mu / 2) * sum(
            loss_values[name] ** 2
            for name, term in self.loss_terms.items()
            if term.is_constraint and term.use_penalty
        )
        loss_values['augmented_term'] = augmented_term.item()

        # Compute total loss
        total_loss = lagrangian_term + augmented_term
        loss_values['loss'] = total_loss.item()

        return total_loss, loss_values

    def update_multipliers(self, loss_values: Dict[str, float]) -> Dict[str, float]:
        """
        Update Lagrange multipliers and penalty parameter.

        Args:
            loss_values: Dictionary of computed loss values

        Returns:
            constraint_violations: Dictionary of constraint violations
        """
        # Compute constraint violations
        constraint_violations = {
            name: term.get_violation(loss_values[name])
            for name, term in self.loss_terms.items()
            if term.is_constraint
        }

        # Update Lagrange multipliers if constraints are violated
        if any(v > 0 for v in constraint_violations.values()):
            for name, term in self.loss_terms.items():
                if term.is_constraint and term.use_lagrange_multiplier:
                    # Create a new tensor for the updated multiplier (avoids in-place operation)
                    new_value = term.lagrange_multiplier.item() + self.mu * loss_values[name]
                    term.lagrange_multiplier = torch.tensor(new_value, requires_grad=True)

            # Update penalty parameter
            max_violation = max(constraint_violations.values(), default=0)
            if self.mu < self.mu_max:
                self.mu = min(self.mu * (1 + max_violation), self.mu_max)

        return constraint_violations


# Example loss term implementations for your PINN problem
def int_loss_fn(model, x, epoch=None):
    """Interior PDE loss"""
    y_pred = model(x)
    y_pred_x = torch.autograd.grad(y_pred, x, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    pde_residual = torch.sum(y_pred_x ** 2, dim=1) - 1.0
    return torch.mean(pde_residual ** 2)


def boundary_loss_fn(model, x):
    """Boundary condition loss"""
    return torch.mean(model(x) ** 2)


def boundary_grad_loss_fn(model, x):
    """Boundary gradient loss"""
    y_pred = model(x)
    y_pred_x = torch.autograd.grad(y_pred, x, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    pde_residual = torch.sum(y_pred_x ** 2, dim=1)
    return torch.mean(pde_residual ** 2)


def dmax_loss_fn(model, x):
    """Maximum distance loss"""
    return torch.exp(-0.5 * torch.mean(model(x)))


def abs_loss_fn(model, x_int, x_b):
    """Absolute value constraint loss"""
    return (torch.mean((model(x_int) - torch.abs(model(x_int))) ** 2) +
            torch.mean((model(x_b) - torch.abs(model(x_b))) ** 2))


def projection_loss_fn(model, x, epoch=None):
    """Projection loss"""
    if not x.requires_grad:
        x.requires_grad_(True)
    u_x = model(x)
    grad_u_x = torch.autograd.grad(outputs=u_x, inputs=x,
                                   grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0]
    projection = x - u_x * grad_u_x
    return torch.mean(model(projection) ** 2)


# Example usage for training a PINN
def train_pinn(model, optimizer, x_train, x_train_0, num_epochs, exp_manager, data_dim, dev, pretrain_optimizer=None, pretrain_epochs=0):
    """
    Train a PINN using the modular augmented Lagrangian method.
    """
    print("Starting training with modular augmented Lagrangian")
    x_train.requires_grad = True
    x_train_0.requires_grad = True

    metrics_collector = MetricsCollector(exp_manager, data_dim, 'train')
    checkpoint = ModelCheckpoint(exp_manager, data_dim=data_dim, max_keep=3)
    early_stopper = EnhancedEarlyStopping()

    # Create loss terms
    loss_terms = [
        # Interior PDE loss
        LossTerm(
            name="loss_int",
            compute_fn=lambda: int_loss_fn(model, x_train, epoch),
            is_constraint=True,
            tolerance=1e-4,
            use_lagrange_multiplier=True,
            use_penalty=True
        ),
        # Boundary condition loss
        LossTerm(
            name="loss_b",
            compute_fn=lambda: boundary_loss_fn(model, x_train_0),
            is_constraint=True,
            tolerance=1e-4,
            use_lagrange_multiplier=True,
            use_penalty=True
        ),
        # Boundary gradient loss
        LossTerm(
            name="loss_grad_boundry",
            compute_fn=lambda: boundary_grad_loss_fn(model, x_train_0),
            is_constraint=True,
            tolerance=1e-4,
            use_lagrange_multiplier=True,
            use_penalty=True
        ),
        # Maximum distance loss (objective term)
        LossTerm(
            name="loss_dmax",
            compute_fn=lambda: dmax_loss_fn(model, x_train),
            is_constraint=False,
            use_lagrange_multiplier=False,
            weight=2.0
        ),
        # Absolute value constraint
        LossTerm(
            name="loss_abs",
            compute_fn=lambda: abs_loss_fn(model, x_train, x_train_0),
            is_constraint=True,
            tolerance=1e-4,
            use_lagrange_multiplier=True,
            use_penalty=True
        ),
        # Projection losses (commented out by default)
        # LossTerm(
        #     name="loss_proj_b",
        #     compute_fn=lambda: projection_loss_fn(model, x_train_0, epoch),
        #     is_constraint=True,
        #     tolerance=1e-4,
        #     use_lagrange_multiplier=True,
        #     use_penalty=True
        # ),
        # LossTerm(
        #     name="loss_proj_int",
        #     compute_fn=lambda: projection_loss_fn(model, x_train, epoch),
        #     is_constraint=True,
        #     tolerance=1e-4,
        #     use_lagrange_multiplier=True,
        #     use_penalty=True
        # )
    ]

    # Initialize augmented Lagrangian manager
    aug_lagrangian = ModularAugmentedLagrangian(
        model=model,
        loss_terms=loss_terms,
        mu_initial=1.0,
        mu_max=1e6
    )

    freq_update = 1

    current_optimizer = pretrain_optimizer if pretrain_optimizer is not None else optimizer

    for epoch in range(num_epochs):
        # Store loss values outside closure for access after optimization step
        current_loss_values = {}

        def closure():
            nonlocal current_loss_values
            optimizer.zero_grad()

            # Compute total loss and get individual loss values
            loss, loss_values = aug_lagrangian.compute_loss()

            # Store loss values for use outside the closure - detach and move to CPU
            current_loss_values = {k: v.detach().cpu().item() if torch.is_tensor(v) else v
                                   for k, v in loss_values.items()}

            # Check for NaN values before backward pass
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}. Stopping training.")
                return loss

            loss.backward()
            return loss

        # Update model parameters - this runs the closure function
        loss = current_optimizer.step(closure)

        if pretrain_optimizer is not None and epoch + 1 == pretrain_epochs:
            current_optimizer = optimizer

        # Use the loss values calculated within the closure
        constraint_violations = aug_lagrangian.update_multipliers(current_loss_values)

        if epoch % freq_update == 0:
            # Calculate accuracy metrics
            y_pred = model(x_train)
            if torch.isnan(y_pred).any():
                print(f"NaN in model output at epoch {epoch}. Stopping training.")
                break

            y_true = distance_function(x_train)
            train_acc_error = torch.mean(torch.abs(y_pred.reshape(-1) - y_true))

            # Save metrics and update checkpoint
            checkpoint.update(model, epoch, current_loss_values['loss'], train_acc_error.item())

            # Add metrics to collector
            metrics_data = {
                'epoch': epoch,
                'accuracy_error': train_acc_error.item(),
                'mu': aug_lagrangian.mu,
                **current_loss_values
            }

            # Add Lagrange multipliers and constraint violations
            for name, term in aug_lagrangian.loss_terms.items():
                if term.use_lagrange_multiplier:
                    metrics_data[f'lambda_{name}'] = term.lagrange_multiplier.item()
                if term.is_constraint:
                    metrics_data[f'{name}_violation'] = constraint_violations.get(name, 0)

            metrics_collector.add_metric(**metrics_data)

            # Check early stopping
            stop_training, best_state = early_stopper.update(
                epoch,
                current_loss_values,
                model.state_dict()
            )

            if stop_training:
                print(f"Early stopping triggered at epoch {epoch}")
                if best_state is not None:
                    best_state = unwrap_state_dict(best_state)
                    if isinstance(model, nn.DataParallel):
                        model.module.load_state_dict(best_state)
                    else:
                        model.load_state_dict(best_state)
                checkpoint.save_best_models('final_model')
                metrics_collector.save_all_metrics()
                metrics_collector.plot_metrics()
                break

            # Print progress
            # Print progress in a single line
            status_line = f'Epoch {epoch}:[{data_dim}], Error: {train_acc_error:.6f}, Loss: {current_loss_values["loss"]:.6f}, '
            status_line += f'Lagrangian-term: {current_loss_values["lagrangian_term"]:.6f}, Augmented-term: {current_loss_values["augmented_term"]:.6f}, '

            # Add loss terms info
            terms_info = []
            for name, term in aug_lagrangian.loss_terms.items():
                weight_val = term.get_weight() if not torch.is_tensor(
                    term.get_weight()) else term.get_weight().item()
                weight_str = f"λ={weight_val:.4f}" if term.use_lagrange_multiplier else f"w={weight_val:.4f}"
                violation_str = f"v={constraint_violations.get(name, 0):.6f}" if term.is_constraint else ""
                terms_info.append(
                    f"{name}={current_loss_values[name]:.6f}({weight_str}{' ' + violation_str if violation_str else ''})")

            status_line += ', '.join(terms_info) + f", μ={aug_lagrangian.mu:.4f}"
            print(status_line)

        # If training loop completes without early stopping, save all metrics
    metrics_collector.save_all_metrics()
    metrics_collector.plot_metrics()
    checkpoint.save_best_models('final_model')

    return model, metrics_collector


def evaluate_model(model, dim, exp_manager, dev):
    """
    Evaluate the trained model and save results.
    """
    model.eval()
    test_range_vec = np.arange(1, 11)
    mean_l1_error = np.zeros(len(test_range_vec))
    std_l1_error = np.zeros(len(test_range_vec))

    # Import ManifoldFactory from your manifolds package.
    from manifolds.manifold_factory import ManifoldFactory
    manifold = ManifoldFactory.get_instance_direct(dev, exp_manager.manifold_type, output_dim=dim)

    metrics_collector = MetricsCollector(exp_manager, dim, 'test')
    for r_ind, rr in enumerate(test_range_vec):
        n_test = 1000
        x_test = sample_x_train_int(n_test, manifold, r=rr, dev=dev).to(dev)
        with torch.no_grad():
            r_test_pred = model(x_test)
            r_test_true = distance_function(x_test)
            l1_error = torch.abs(r_test_true - r_test_pred.reshape(-1))
            mean_l1_error[r_ind] = torch.mean(l1_error).item()
            std_l1_error[r_ind] = torch.std(l1_error).item()
            metrics_collector.add_metric(
                epoch=r_ind,
                test_range=rr,
                mean_error=mean_l1_error[r_ind],
                std_error=std_l1_error[r_ind]
            )
    plt.figure(figsize=(10, 6))
    plt.errorbar(test_range_vec, mean_l1_error, yerr=std_l1_error,
                 fmt='-o', capsize=5, capthick=2, ecolor='red')
    plt.xlabel('Test Range')
    plt.ylabel('Mean L1 Error')
    plt.title(f'Mean L1 Error with Standard Deviation (Dimension {dim})')
    plt.grid(True)
    plt.savefig(exp_manager.get_path('plots', f'error_plot_dim_{dim}.png'))
    plt.close()
    np.savez(exp_manager.get_path('results', f'results_dim_{dim}.npz'),
             mean_error=mean_l1_error,
             std_error=std_l1_error,
             test_range=test_range_vec)
    return mean_l1_error, std_l1_error