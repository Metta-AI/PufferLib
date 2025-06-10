import functools
import numpy as np
import gymnasium

import pufferlib

from mettagrid.curriculum import SingleTaskCurriculum

def env_creator(name='metta'):
    return functools.partial(make, name)

def make(name, config='pufferlib/environments/metta/metta.yaml', render_mode='auto', buf=None, seed=0,
         ore_reward=0.25, heart_reward=0.5, battery_reward=0.25):
    '''Crafter creation function'''
    #return MettaPuff(config, render_mode, buf)
    import mettagrid.mettagrid_env
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("div", oc_divide, replace=True)
    cfg = OmegaConf.load(config)
    reward_cfg = cfg['game']['agent']['rewards']
    '''
    env_overrides = {
        'game': {
            'agent': {
                'rewards': {
                    'ore.red': 0.25,
                    'ore.blue': 0.25,
                    'ore.green': 0.25,
                    'heart': 0.5,
                    'battery': 0.25,
                }
            }
        }
    '''
    reward_cfg['ore.red'] = ore_reward
    reward_cfg['ore.blue'] = ore_reward
    reward_cfg['ore.green'] = ore_reward
    reward_cfg['heart'] = heart_reward
    reward_cfg['battery'] = battery_reward
    curriculum = SingleTaskCurriculum('puffer', cfg)
    return MettaPuff(curriculum, render_mode=render_mode, buf=buf)

def oc_divide(a, b):
    """
    Divide a by b, returning an int if both inputs are ints and result is a whole number,
    otherwise return a float.
    """
    result = a / b
    # If both inputs are integers and the result is a whole number, return as int
    if isinstance(a, int) and isinstance(b, int) and result.is_integer():
        return int(result)
    return result

class FlattenedDiscrete(gymnasium.spaces.Discrete):
    """A Discrete action space that maintains compatibility with MultiDiscrete validation.
    
    This class extends gymnasium's Discrete space to provide an `nvec` property
    that MettaGrid's validation code expects, while still functioning as a 
    standard Discrete space for PufferLib.
    """
    
    def __init__(self, n, original_nvec, seed=None):
        super().__init__(n, seed=seed)
        self._original_nvec = np.array(original_nvec, dtype=np.int64)
    
    @property
    def nvec(self):
        """Provide nvec for backward compatibility with MettaGrid validation."""
        return self._original_nvec

class MettaActionAdapter:
    """Adapter to convert between flat discrete actions and MettaGrid's MultiDiscrete format."""
    
    def __init__(self, env):
        """Initialize the adapter with a MettaGrid environment."""
        self.max_action_args = env.max_action_args
        self.action_names = env.action_names
        
        # Build the flattened action space
        self.arg_counts = [max_arg + 1 for max_arg in self.max_action_args]
        self.n_actions = sum(self.arg_counts)
        
        # Create mapping from flat index to (action_type, action_arg)
        self.action_map = np.zeros((self.n_actions, 2), dtype=np.int32)
        
        i = 0
        for action_type, (action_name, arg_count) in enumerate(zip(self.action_names, self.arg_counts)):
            for arg in range(arg_count):
                self.action_map[i] = (action_type, arg)
                i += 1
    
    def get_flat_action_space(self):
        """Get the flattened discrete action space."""
        # Create FlattenedDiscrete with original nvec for compatibility
        original_nvec = [len(self.action_names)] + self.max_action_args
        return FlattenedDiscrete(self.n_actions, original_nvec)
    
    def unflatten_from_discrete(self, flat_action):
        """Convert flat discrete action to (action_type, action_arg)."""
        if isinstance(flat_action, (list, np.ndarray)):
            # Handle batched actions
            return np.array([self.action_map[a] for a in flat_action])
        else:
            # Single action
            return self.action_map[flat_action]

class MettaPuff(MettaGridEnv):
    def __init__(self, config, render_mode='human', buf=None, seed=0):
        super().__init__(config, render_mode=render_mode, buf=buf)
        self._action_adapter = MettaActionAdapter(self)
        self._flattened_action_space = self._action_adapter.get_flat_action_space()
        self.action_space = pufferlib.spaces.joint_space(self._flattened_action_space, self.num_agents)
        self.actions = self.actions.astype(np.int32)
    
    @property
    def single_action_space(self):
        """Return flattened single action space for PufferLib."""
        if hasattr(self, '_flattened_action_space'):
            return self._flattened_action_space
        return super().single_action_space
    
    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int32)
        
        if actions.ndim == 1 and len(actions) == self.num_agents:
            unflattened_actions = np.array([
                self._action_adapter.unflatten_from_discrete(a) 
                for a in actions
            ], dtype=np.int32)
        elif actions.ndim == 1 and len(actions) == 1:
            # Single action to broadcast to all agents
            unflattened = self._action_adapter.unflatten_from_discrete(actions[0])
            unflattened_actions = np.tile(unflattened, (self.num_agents, 1))
        elif actions.ndim == 2 and actions.shape[0] == self.num_agents:
            # Already shaped for agents
            if actions.shape[1] == 1:
                # Single flat action per agent
                unflattened_actions = np.array([
                    self._action_adapter.unflatten_from_discrete(a[0]) 
                    for a in actions
                ], dtype=np.int32)
            else:
                # Assume already in (action_type, action_arg) format
                unflattened_actions = actions
        else:
            raise ValueError(f"Invalid action shape: {actions.shape}. Expected (num_agents,) or (num_agents, 1)")
        
        obs, rew, term, trunc, info = super().step(unflattened_actions)
        
        if all(term) or all(trunc):
            self.reset()
            if 'agent_raw' in info:
                del info['agent_raw']
            if 'episode_rewards' in info:
                info['score'] = info['episode_rewards']
        else:
            info = []
        
        return obs, rew, term, trunc, [info]
