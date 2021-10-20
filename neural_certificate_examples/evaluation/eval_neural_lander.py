import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from neural_certificate_examples.controllers import NeuralCLFController


def eval_neural_lander():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/neural_lander/"
    neural_controller = NeuralCLFController.load_from_checkpoint(log_dir + "clf.ckpt")

    # Increase the duration of the simulation
    neural_controller.experiment_suite.experiments[1].t_sim = 10.0

    # Run the experiments and save the results
    neural_controller.experiment_suite.run_all_and_save_to_csv(
        neural_controller, log_dir + "experiments"
    )


def plot_neural_lander():
    # Load the checkpoint file. This should include the experiment suite used during
    # training.
    log_dir = "saved_models/neural_lander/"
    neural_controller = NeuralCLFController.load_from_checkpoint(log_dir + "clf.ckpt")

    # Set the path to load from
    experiment_dir = log_dir + "experiments/2021-10-20_14_16_16"

    # Load the data from the simulation
    results_df = pd.read_csv(experiment_dir + "/Rollout.csv")

    # Plot the simulation results
    sns.set_theme(context="talk", style="white")
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(10, 4)
    z_trace = results_df[results_df["measurement"] == "$z$"]
    sns.lineplot(
        ax=axs,
        x=z_trace["t"],
        y=z_trace["value"],
    )
    axs.set_ylabel("$z$")
    axs.set_xlabel("t")
    plt.show()


if __name__ == "__main__":
    # eval_neural_lander()
    plot_neural_lander()
