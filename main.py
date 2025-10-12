import gymnasium as gym
import optuna
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_learning_curve(log_path, window_size=100):
    """Generates and saves a plot of the learning curve."""
    try:
        log_data = pd.read_csv(log_path + ".monitor.csv", skiprows=1)
        cumulative_timesteps = log_data['l'].cumsum()
        moving_avg = log_data['r'].rolling(window=window_size).mean()
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_timesteps, log_data['r'], alpha=0.3, label='Per-Episode Reward')
        plt.plot(cumulative_timesteps, moving_avg, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
        plt.title("Learning Curve of the Final Agent")
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig("learning_curve.png")
        print("\nLearning curve plot saved to learning_curve.png")
    except FileNotFoundError:
        print("\nCould not find monitor log file. Skipping learning curve plot.")

def plot_optuna_study(study):
    """Generates and saves a plot of the Optuna study."""
    if not study.trials:
        print("No trials found in the study to plot.")
        return
    trial_values = [t.value for t in study.trials if t.value is not None]
    if not trial_values:
        print("No successful trials with values to plot.")
        return
    trial_numbers = [t.number for t in study.trials if t.value is not None]
    plt.figure(figsize=(10, 6))
    plt.plot(trial_numbers, trial_values, marker='o', linestyle='--')
    if study.best_trial and study.best_trial.value is not None:
        best_trial_num = study.best_trial.number
        best_trial_val = study.best_trial.value
        plt.scatter(best_trial_num, best_trial_val, s=120, c='red', zorder=5, label=f'Best Trial (#{best_trial_num})')
    plt.title("Optuna Hyperparameter Optimization History")
    plt.xlabel("Trial Number")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("optuna_study.png")
    print("Optuna study plot saved to optuna_study.png")
    plt.close()

# Hyperparameter tuning with Optuna
def objective(trial):
    env = Monitor(gym.make("LunarLander-v3"))

    # Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    layer_size = trial.suggest_categorical("layer_size", [128, 256, 512])
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    buffer_size = trial.suggest_categorical("buffer_size", [50000, 100000])
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.2, 0.4)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.05)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1000, 5000])
    train_freq = trial.suggest_categorical("train_freq", [4, 8])
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 4])
    learning_starts = trial.suggest_categorical("learning_starts", [1000, 5000])

    net_arch = [layer_size, layer_size]
    policy_kwargs = dict(net_arch=net_arch)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        learning_starts=learning_starts,
        policy_kwargs=policy_kwargs,
        verbose=0
    )

    model.learn(total_timesteps=25000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=20)
    env.close()
    return mean_reward

if __name__ == "__main__":
    # Check and create logs
    LOG_DIR = "logs/"
    os.makedirs(LOG_DIR, exist_ok=True)

    # Runtime variables
    STORAGE_PATH = "sqlite:///my_study.db" # database
    STUDY_NAME = "lunarlander-optimization" # Name of your study
    NUM_TRIALS_TO_RUN = 30 # Number of trial runs

    # Remove bad runs early
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True,
        pruner=pruner
    )

    # Avoid additional trials | Increase variable for more runs
    if len(study.trials) < NUM_TRIALS_TO_RUN:
        study.optimize(objective, n_trials=NUM_TRIALS_TO_RUN - len(study.trials))
    else:
        print(f"Study already has {len(study.trials)} trials. Skipping optimization.")

    print("\n--- Best Trial Information ---")
    best_trial = study.best_trial
    if best_trial:
        print(f"  Value (Mean Reward): {best_trial.value:.2f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        print("\n--- Training the final, best model ---")
        best_params = best_trial.params.copy()
        final_layer_size = best_params.pop('layer_size')
        final_policy_kwargs = dict(net_arch=[final_layer_size, final_layer_size])

        # Create and wrap the environment with Monitor for logging
        final_env = gym.make("LunarLander-v3")
        final_log_path = os.path.join(LOG_DIR, "final_model_logs")
        final_env = Monitor(final_env, final_log_path)

        final_model = DQN("MlpPolicy", final_env, policy_kwargs=final_policy_kwargs,
                          **best_params, verbose=0)

        final_model.learn(total_timesteps=500_000)
        final_model.save("best_lunarlander_model")
        print("\nFinal model saved to best_lunarlander_model.zip")

        # Built-in evaluation
        print("\n--- Evaluating Final Model Performance ---")
        eval_env = gym.make("LunarLander-v3")
        mean_reward, std_reward = evaluate_policy(final_model, eval_env, n_eval_episodes=100)
        print(f"Final Model: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        eval_env.close()

        # Images & Plotting
        plot_optuna_study(study)
        plot_learning_curve(final_log_path)

    else:
        print("No successful trials were completed. Cannot train or evaluate a final model.")