from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_certificate_examples.controllers import NeuralCLBFController
from neural_certificate_examples.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_certificate_examples.systems import NeuralLander
from neural_certificate_examples.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutTimeSeriesExperiment,
)
from neural_certificate_examples.training.utils import current_git_hash


torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.01

start_x = torch.tensor(
    [
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
    ]
)
simulation_dt = 0.001


def main(args):
    # Define the dynamics model
    scenarios = [{}]
    dynamics_model = NeuralLander(
        dt=simulation_dt,
        controller_dt=controller_period,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-5.0, 5.0),  # x
        (-5.0, 5.0),  # y
        (-0.5, 5.0),  # z
        (-1.0, 1.0),  # vx
        (-1.0, 1.0),  # vy
        (-1.0, 1.0),  # vz
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,  # Get all points from sampling, not trajectories
        trajectory_length=1,
        fixed_samples=10000,
        max_points=300000,
        val_split=0.1,
        batch_size=1024,
        quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.4},
    )

    # Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-5.0, 5.0), (-0.5, 2.0)],
        n_grid=20,
        x_axis_index=NeuralLander.PX,
        y_axis_index=NeuralLander.PZ,
        x_axis_label="$x$",
        y_axis_label="$z$",
        plot_unsafe_region=True,
    )
    rollout_experiment = RolloutTimeSeriesExperiment(
        "Rollout",
        start_x,
        plot_x_indices=[NeuralLander.PZ, NeuralLander.VZ],
        plot_x_labels=["$z$", "$\\dot{z}$"],
        plot_u_indices=[],
        plot_u_labels=[],
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])

    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        clbf_hidden_layers=1,
        clbf_hidden_size=48,
        clf_lambda=0.1,
        safe_level=10.0,
        controller_period=controller_period,
        clf_relaxation_penalty=100.0,
        primal_learning_rate=1e-3,
        initial_loss_weight=1.0,
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/neural_lander",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, reload_dataloaders_every_epoch=True
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
