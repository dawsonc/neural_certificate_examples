import itertools
from typing import Tuple, List, Optional
from collections import OrderedDict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from neural_certificate_examples.systems import ControlAffineSystem
from neural_certificate_examples.systems.utils import ScenarioList
from neural_certificate_examples.controllers.clf_controller import CLFController
from neural_certificate_examples.controllers.controller_utils import (
    normalize_with_angles,
)
from neural_certificate_examples.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_certificate_examples.experiments import ExperimentSuite


class NeuralCLFController(pl.LightningModule, CLFController):
    """
    A neural CLF controller. Differs from the CLFController in that it uses a
    neural network to learn the CLF.

    More specifically, the CLF controller looks for a V such that

    V(goal) = 0
    V(not goal) > 0
    dV/dt <= -lambda V

    This proves convergence to the goal
    """

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        scenarios: ScenarioList,
        datamodule: EpisodicDataModule,
        experiment_suite: ExperimentSuite,
        clf_hidden_layers: int = 2,
        clf_hidden_size: int = 48,
        clf_lambda: float = 1.0,
        clf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
        primal_learning_rate: float = 1e-3,
        scale_parameter: float = 10.0,
    ):
        """Initialize the controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            scenarios: a list of parameter scenarios to train on
            experiment_suite: defines the experiments to run during training
            clf_hidden_layers: number of hidden layers to use for the CLF network
            clf_hidden_size: number of neurons per hidden layer in the CLF network
            clf_lambda: convergence rate for the CLF
            clf_relaxation_penalty: the penalty for relaxing CLF conditions.
            controller_period: the timestep to use in simulating forward Vdot
            primal_learning_rate: the learning rate for SGD for the network weights,
                                  applied to the CLF decrease loss
            scale_parameter: normalize non-angle data points to between +/- this value.
        """
        super(NeuralCLFController, self).__init__(
            dynamics_model=dynamics_model,
            scenarios=scenarios,
            experiment_suite=experiment_suite,
            clf_lambda=clf_lambda,
            clf_relaxation_penalty=clf_relaxation_penalty,
            controller_period=controller_period,
        )
        self.save_hyperparameters()

        # Save the provided model
        # self.dynamics_model = dynamics_model
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        # Save the datamodule
        self.datamodule = datamodule

        # Save the experiments suits
        self.experiment_suite = experiment_suite

        # Save the other parameters
        self.primal_learning_rate = primal_learning_rate

        # Compute and save the center and range of the state variables
        x_max, x_min = dynamics_model.state_limits
        self.x_center = (x_max + x_min) / 2.0
        self.x_range = (x_max - x_min) / 2.0
        # Scale to get the input between (-k, k), centered at 0
        self.k = scale_parameter
        self.x_range = self.x_range / self.k
        # We shouldn't scale or offset any angle dimensions
        self.x_center[self.dynamics_model.angle_dims] = 0.0
        self.x_range[self.dynamics_model.angle_dims] = 1.0

        # Some of the dimensions might represent angles. We want to replace these
        # dimensions with two dimensions: sin and cos of the angle. To do this, we need
        # to figure out how many numbers are in the expanded state
        n_angles = len(self.dynamics_model.angle_dims)
        self.n_dims_extended = self.dynamics_model.n_dims + n_angles

        # Define the CLF network, which we denote V
        self.clf_hidden_layers = clf_hidden_layers
        self.clf_hidden_size = clf_hidden_size
        # We're going to build the network up layer by layer, starting with the input
        self.V_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.V_layers["input_linear"] = nn.Linear(
            self.n_dims_extended, self.clf_hidden_size
        )
        self.V_layers["input_activation"] = nn.Tanh()
        for i in range(self.clf_hidden_layers):
            self.V_layers[f"layer_{i}_linear"] = nn.Linear(
                self.clf_hidden_size, self.clf_hidden_size
            )
            if i < self.clf_hidden_layers - 1:
                self.V_layers[f"layer_{i}_activation"] = nn.Tanh()
        self.V_layers["output_linear"] = nn.Linear(self.clf_hidden_size, 1)
        self.V_nn = nn.Sequential(self.V_layers)

    def prepare_data(self):
        return self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None):
        return self.datamodule.setup(stage)

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def V_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the CLF value and its Jacobian

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLF
        returns:
            V: bs tensor of CLF values
            JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        # Apply the offset and range to normalize about zero
        x_norm = normalize_with_angles(self.dynamics_model, x, k=self.k)

        # Compute the CLF layer-by-layer, computing the Jacobian alongside

        # We need to initialize the Jacobian to reflect the normalization that's already
        # been done to x
        bs = x_norm.shape[0]
        JV = torch.zeros(
            (bs, self.n_dims_extended, self.dynamics_model.n_dims)
        ).type_as(x)
        # and for each non-angle dimension, we need to scale by the normalization
        for dim in range(self.dynamics_model.n_dims):
            JV[:, dim, dim] = 1.0 / self.x_range[dim].type_as(x)

        # And adjust the Jacobian for the angle dimensions
        for offset, sin_idx in enumerate(self.dynamics_model.angle_dims):
            cos_idx = self.dynamics_model.n_dims + offset
            JV[:, sin_idx, sin_idx] = x_norm[:, cos_idx]
            JV[:, cos_idx, sin_idx] = -x_norm[:, sin_idx]

        # Now step through each layer in V
        V = x_norm
        for layer in self.V_nn:
            V = layer(V)

            if isinstance(layer, nn.Linear):
                JV = torch.matmul(layer.weight, JV)
            elif isinstance(layer, nn.Tanh):
                JV = torch.matmul(torch.diag_embed(1 - V ** 2), JV)
            elif isinstance(layer, nn.ReLU):
                JV = torch.matmul(torch.diag_embed(torch.sign(V)), JV)

        # Add the nominal CLF
        P = self.dynamics_model.P.type_as(x)
        P = P.reshape(1, self.dynamics_model.n_dims, self.dynamics_model.n_dims)
        x_goal = self.dynamics_model.goal_point.type_as(x)
        V = V + 0.5 * F.bilinear(x - x_goal, x - x_goal, P)
        P = P.reshape(self.dynamics_model.n_dims, self.dynamics_model.n_dims)
        JV = JV + F.linear(x - x_goal, P).reshape(
            x.shape[0], 1, self.dynamics_model.n_dims
        )

        return V, JV

    def forward(self, x):
        """Determine the control input for a given state using a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
        """
        return self.u(x)

    def boundary_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLF due to boundary conditions

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        eps = 1e-2
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []

        V = self.V(x)

        #   1.) V = 0 at the goal
        V_goal = self.V(self.dynamics_model.goal_point)
        loss.append(("CLF goal term", 1e2 * (V_goal ** 2).sum()))

        #   2.) V > 0 elsewhere
        psd_violation = F.relu(eps - V)
        psd_V_term = psd_violation.mean()
        loss.append(("CLF PSD term", psd_V_term))
        if accuracy:
            psd_V_acc = (psd_violation <= eps).sum() / psd_violation.nelement()
            loss.append(("CLF PSD accuracy", psd_V_acc))

        #   3.) V > x^T P x
        P = self.dynamics_model.P.type_as(x)
        P = torch.eye(self.dynamics_model.n_dims)
        P = P.reshape(1, self.dynamics_model.n_dims, self.dynamics_model.n_dims)
        tuning_violation = F.relu(0.1 * F.bilinear(x, x, P) - V)
        tuning_V_term = 1e-1 * tuning_violation.mean()
        loss.append(("CLF tuning term", tuning_V_term))
        if accuracy:
            tuning_V_acc = (tuning_violation <= eps).sum() / tuning_violation.nelement()
            loss.append(("CLF tuning accuracy", tuning_V_acc))

        return loss

    def descent_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLF due to the descent condition

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []

        # We'll encourage satisfying the CBF conditions by...
        #
        #   1) Minimize the relaxation needed to make the QP feasible.

        # Get the control input and relaxation from solving the QP, and aggregate
        # the relaxation across scenarios
        u_qp, qp_relaxation = self.solve_CLF_QP(x)
        qp_relaxation = torch.mean(qp_relaxation, dim=-1)

        # Minimize the qp relaxation to encourage satisfying the decrease condition
        qp_relaxation_loss = qp_relaxation.mean()
        loss.append(("QP relaxation", qp_relaxation_loss))

        return loss

    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask = batch

        # Compute the losses
        component_losses = {}
        component_losses.update(
            self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask)
        )
        component_losses.update(self.descent_loss(x, goal_mask, safe_mask, unsafe_mask))

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        batch_dict = {"loss": total_loss, **component_losses}

        return batch_dict

    def training_epoch_end(self, outputs):
        """This function is called after every epoch is completed."""
        # Outputs contains a list for each optimizer, and we need to collect the losses
        # from all of them if there is a nested list
        if isinstance(outputs[0], list):
            outputs = itertools.chain(*outputs)

        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in outputs:
            for key in batch_output.keys():
                # if we've seen this key before, add this component loss to the list
                if key in losses:
                    losses[key].append(batch_output[key])
                else:
                    # otherwise, make a new list
                    losses[key] = [batch_output[key]]

        # Average all the losses
        avg_losses = {}
        for key in losses.keys():
            key_losses = torch.stack(losses[key])
            avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

        # Log the overall loss...
        self.log("Total loss / train", avg_losses["loss"], sync_dist=True)
        # And all component losses
        for loss_key in avg_losses.keys():
            # We already logged overall loss, so skip that here
            if loss_key == "loss":
                continue
            # Log the other losses
            self.log(loss_key + " / train", avg_losses[loss_key], sync_dist=True)

    def validation_step(self, batch, batch_idx):
        """Conduct the validation step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask = batch

        # Get the various losses
        component_losses = {}
        component_losses.update(
            self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask)
        )
        component_losses.update(self.descent_loss(x, goal_mask, safe_mask, unsafe_mask))

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        # Also compute the accuracy associated with each loss
        component_losses.update(
            self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask, accuracy=True)
        )
        component_losses.update(
            self.descent_loss(x, goal_mask, safe_mask, unsafe_mask, accuracy=True)
        )

        batch_dict = {"val_loss": total_loss, **component_losses}

        return batch_dict

    def validation_epoch_end(self, outputs):
        """This function is called after every epoch is completed."""
        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in outputs:
            for key in batch_output.keys():
                # if we've seen this key before, add this component loss to the list
                if key in losses:
                    losses[key].append(batch_output[key])
                else:
                    # otherwise, make a new list
                    losses[key] = [batch_output[key]]

        # Average all the losses
        avg_losses = {}
        for key in losses.keys():
            key_losses = torch.stack(losses[key])
            avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

        # Log the overall loss...
        self.log("Total loss / val", avg_losses["val_loss"], sync_dist=True)
        # And all component losses
        for loss_key in avg_losses.keys():
            # We already logged overall loss, so skip that here
            if loss_key == "val_loss":
                continue
            # Log the other losses
            self.log(loss_key + " / val", avg_losses[loss_key], sync_dist=True)

        # **Now entering spicetacular automation zone**
        # We automatically run experiments every few epochs

        # # Only plot every 5 epochs
        # if self.current_epoch % 5 != 0:
        #     return

        self.experiment_suite.run_all_and_log_plots(
            self, self.logger, self.current_epoch
        )

    @pl.core.decorators.auto_move_data
    def simulator_fn(
        self,
        x_init: torch.Tensor,
        num_steps: int,
        relaxation_penalty: Optional[float] = None,
    ):
        # Choose parameters randomly
        random_scenario = {}
        for param_name in self.scenarios[0].keys():
            param_max = max([s[param_name] for s in self.scenarios])
            param_min = min([s[param_name] for s in self.scenarios])
            random_scenario[param_name] = random.uniform(param_min, param_max)

        return self.dynamics_model.simulate(
            x_init,
            num_steps,
            self.u,
            guard=self.dynamics_model.out_of_bounds_mask,
            controller_period=self.controller_period,
            params=random_scenario,
        )

    def configure_optimizers(self):
        clf_params = list(self.V_nn.parameters())

        clf_opt = torch.optim.SGD(
            clf_params,
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )

        self.opt_idx_dict = {0: "clf"}

        return [clf_opt]
