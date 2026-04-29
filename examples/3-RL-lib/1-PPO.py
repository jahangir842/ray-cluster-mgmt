import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from pprint import pprint

# 1. Initialize Ray (if not already running)
ray.init(ignore_reinit_error=True)

# 2. Configure the algorithm
# We set num_env_runners to 2 as per your request to parallelize data collection.
config = (
    PPOConfig()
    .environment("Taxi-v3")
    .env_runners(
        num_env_runners=2,
        # Taxi-v3 returns discrete integer observations. 
        # FlattenObservations converts these to a one-hot encoded format 
        # that the PPO neural network can process.
        env_to_module_connector=lambda env: FlattenObservations(),
    )
    .evaluation(evaluation_num_env_runners=1)
)

# 3. Build the algorithm
print("Building Algorithm...")
algo = config.build_algo()

# 4. Train for 5 iterations
# Each iteration will trigger the EnvRunners to collect data, 
# followed by the PPO policy update.
print("Starting Training...")
for i in range(5):
    print(f"\n--- Iteration {i+1} ---")
    result = algo.train()
    # Printing the result dictionary (contains episode reward, entropy, etc.)
    pprint(result["env_runners"]) 

# 5. Evaluate and Cleanup
print("\n--- Evaluation ---")
pprint(algo.evaluate())

# Properly shut down the remote actors
algo.stop()
ray.shutdown()
print("\nDone.")