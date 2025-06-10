import functools
import numpy as np
import gymnasium

import pufferlib

from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.curriculum import SingleTaskCurriculum

import functools
import numpy as np
import gymnasium
from gymnasium.spaces import Discrete
import pufferlib

from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.curriculum import SingleTaskCurriculum

class MettaActionAdapter:
    """Adapter to convert between flat discrete actions and MettaGrid's MultiDiscrete format."""
    
    def __init__(self, env):
        """Initialize the adapter with a MettaGrid environment."""
        # Get the action space info
        self.max_action_args = env.max_action_args
        self.action_names = env.action_names
        
        # Build the flattened action space
        self.arg_counts = [max_arg + 1 for max_arg in self.max_action_args]
        self.n_actions = sum(self.arg_counts)
        
        # Create mapping from flat index to (action_type, action_arg)
        self.action_map = np.zeros((self.n_actions, 2), dtype=np.int32)
        
        # Create reverse mapping for debugging
        self.reverse_map = {}
        self.flat_to_name = {}
        
        i = 0
        for action_type, (action_name, arg_count) in enumerate(zip(self.action_names, self.arg_counts)):
            for arg in range(arg_count):
                self.action_map[i] = (action_type, arg)
                self.reverse_map[i] = f"{action_name}({arg})"
                self.flat_to_name[i] = (action_name, arg)
                i += 1
        
        # Store action type to name mapping
        self.action_type_to_name = {i: name for i, name in enumerate(self.action_names)}
        self.action_name_to_type = {name: i for i, name in enumerate(self.action_names)}
    
    def get_flat_action_space(self):
        """Get the flattened discrete action space."""
        return Discrete(self.n_actions)
    
    def unflatten_from_discrete(self, flat_action):
        """Convert flat discrete action to (action_type, action_arg)."""
        if isinstance(flat_action, (list, np.ndarray)):
            # Handle batched actions
            return np.array([self.action_map[a] for a in flat_action])
        else:
            # Single action
            return self.action_map[flat_action]
    
    def get_action_description(self, flat_action):
        """Get human-readable description of action."""
        return self.reverse_map.get(flat_action, f"unknown_action_{flat_action}")
    
    def print_action_space_info(self):
        """Print detailed information about the action space."""
        print(f"Total flattened actions: {self.n_actions}")
        print("\nAction types and their arguments:")
        for i, (name, max_arg) in enumerate(zip(self.action_names, self.max_action_args)):
            arg_count = max_arg + 1
            print(f"  {i}: {name} - args: 0-{max_arg} ({arg_count} options)")
        
        print(f"\nFlattened action space: Discrete({self.n_actions})")


class MettaPuffWrapper(gymnasium.Wrapper):
    """Wrapper that converts MettaGrid's MultiDiscrete action space to flat Discrete."""
    
    def __init__(self, env):
        super().__init__(env)
        self.action_adapter = MettaActionAdapter(env)
        self.num_agents = 24
        
        # Override action spaces
        self._single_action_space = self.action_adapter.get_flat_action_space()
        self._action_space = pufferlib.spaces.joint_space(
            self._single_action_space, 
            self.num_agents
        )
    
    @property
    def single_action_space(self):
        return self._single_action_space
    
    @property
    def action_space(self):
        return self._action_space
    
    def step(self, actions):
        # Convert flat discrete actions to MultiDiscrete format
        actions = np.asarray(actions, dtype=np.int32)
        
        # Handle different input shapes
        if actions.ndim == 1 and len(actions) == self.num_agents:
            # Flat actions for each agent
            unflattened_actions = np.array([
                self.action_adapter.unflatten_from_discrete(a) 
                for a in actions
            ], dtype=np.int32)
        elif actions.ndim == 1 and len(actions) == 1:
            # Single action to broadcast to all agents
            unflattened = self.action_adapter.unflatten_from_discrete(actions[0])
            unflattened_actions = np.tile(unflattened, (self.num_agents, 1))
        elif actions.ndim == 2 and actions.shape[0] == self.num_agents:
            # Already shaped for agents
            if actions.shape[1] == 1:
                # Single flat action per agent
                unflattened_actions = np.array([
                    self.action_adapter.unflatten_from_discrete(a[0]) 
                    for a in actions
                ], dtype=np.int32)
            else:
                # Assume already in (action_type, action_arg) format
                unflattened_actions = actions
        else:
            raise ValueError(f"Invalid action shape: {actions.shape}. Expected (num_agents,) or (num_agents, 1)")
        
        # Call wrapped environment with unflattened actions
        obs, rew, term, trunc, info = self.env.step(unflattened_actions)
        
        # Post-process for pufferlib compatibility
        if all(term) or all(trunc):
            self.env.reset()
            if isinstance(info, dict):
                if 'agent_raw' in info:
                    del info['agent_raw']
                if 'episode_rewards' in info:
                    info['score'] = info['episode_rewards']
                info = [info]
        else:
            if not isinstance(info, list):
                info = []
        
        return obs, rew, term, trunc, info


# Factory functions
def oc_divide(a, b):
    """Divide a by b, returning an int if both inputs are ints and result is a whole number."""
    result = a / b
    if isinstance(a, int) and isinstance(b, int) and result.is_integer():
        return int(result)
    return result

def env_creator(name='metta'):
    return functools.partial(make, name)

def make(name, config='pufferlib/environments/metta/metta.yaml', render_mode='auto', buf=None, seed=0,
         ore_reward=0.25, heart_reward=0.5, battery_reward=0.25):
    '''Metta creation function with flattened action space'''
    from omegaconf import OmegaConf
    
    OmegaConf.register_new_resolver("div", oc_divide, replace=True)
    cfg = OmegaConf.load(config)
    
    # Modify rewards
    reward_cfg = cfg['game']['agent']['rewards']
    reward_cfg['ore.red'] = ore_reward
    reward_cfg['ore.blue'] = ore_reward
    reward_cfg['ore.green'] = ore_reward
    reward_cfg['heart'] = heart_reward
    reward_cfg['battery'] = battery_reward
    
    # Create curriculum
    curriculum = SingleTaskCurriculum('puffer', cfg)
    
    # Create base environment
    base_env = MettaGridEnv(curriculum, render_mode=render_mode, buf=buf)
    
    # Wrap with action adapter
    return MettaPuffWrapper(base_env)
