from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules import (
    EpisodicDataModule,
)
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutTimeSeriesExperiment,
)
from neural_clbf.systems import TurtleBot


torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.01

start_x = torch.tensor(
    [
        [0, 0, 0],
        # [1.0, 1.0, 0],
    ]
)
simulation_dt = 0.001


def main(args):
    # Define the scenarios
    nominal_params = {"R": 3.25, "L": 14.0}
    scenarios = [
        nominal_params,
        # {"R": 3.25, "L": 13.9},
        # {"R": 3.25, "L": 14.1},
    ]

    # Define the dynamics model
    dynamics_model = TurtleBot(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-2.0, 2.0),  # x
        (-2.0, 2.0),  # y
        (-np.pi / 2, np.pi / 2),  # theta
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=100,
        trajectory_length=500,
        fixed_samples=20000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
        quotas={"safe": 0.2, "unsafe": 0.2, "goal": 0.4},
    )

    # Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V Contour",
        domain=[(-2.0, 2.0), (-2.0, 2.0)],
        n_grid=50,
        x_axis_index=TurtleBot.X,
        y_axis_index=TurtleBot.Y,
        x_axis_label="$x$",
        y_axis_label="$y$",
    )
    rollout_experiment = RolloutTimeSeriesExperiment(
        "Rollout",
        start_x,
        plot_x_indices=[TurtleBot.X, TurtleBot.Y, TurtleBot.THETA],
        plot_x_labels=["$x$", "$y$"],
        plot_u_indices=[TurtleBot.V, TurtleBot.THETA_DOT],
        plot_u_labels=["$v$", "$\\dot{\\theta}$"],
        t_sim=6.0,
        n_sims_per_start=2,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])

    # Initialize the controller
    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite,
        clbf_hidden_layers=2,
        clbf_hidden_size=64,
        u_nn_hidden_layers=2,
        u_nn_hidden_size=64,
        clf_lambda=1.0,
        safe_level=1.0,
        controller_period=controller_period,
        clf_relaxation_penalty=1e5,
        num_init_epochs=5,
        epochs_per_episode=100,
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/turtlebot",
        name="full_test",
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