from uqslam.p3at_slow_control_env import P3atSlowControlEnv
from gymnasium.vector import SyncVectorEnv
from gymnasium.spaces import Space

class P3atVecEnv(SyncVectorEnv):
    """
    Vectorized wrapper for multiple P3atSlowControlEnv environments.
    Uses Gymnasium's SyncVectorEnv for synchronous multi-agent simulation.
    """
    def __init__(self, prefix="robot", repeat_steps=4, num_envs=1):
        # Create a list of callables that return new env instances
        env_fns = [
            lambda i=i: P3atSlowControlEnv(
                prefix=f"{prefix}{i+1}",
                robot_namespace=f"{prefix}{i+1}_ns",
                repeat_steps=repeat_steps
            ) for i in range(num_envs)
        ]

        # Call the parent constructor with these env factories
        super().__init__(env_fns)

        # Cache action/obs spaces for convenience
        self.action_space: Space = self.single_action_space
        self.observation_space: Space = self.single_observation_space
