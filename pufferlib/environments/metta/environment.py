import functools
import numpy as np
import gymnasium

import pufferlib

from mettagrid.mettagrid_env import MettaGridEnv
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
    cfg = SingleTaskCurriculum('puffer', cfg)
    return MettaPuff(cfg, render_mode=render_mode, buf=buf)

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
    def __init__(self, n, original_nvec, seed=None):
        super().__init__(n, seed=seed)
        self._original_nvec = np.array(original_nvec, dtype=np.int64)
    
    @property
    def nvec(self):
        """Provide nvec for backward compatibility with MettaGrid validation."""
        return self._original_nvec
class MettaPuff(MettaGridEnv):
    def __init__(self, config, render_mode='human', buf=None, seed=0):
        super().__init__(config, render_mode=render_mode, buf=buf)
        
        # Build flattened action mapping
        self._build_action_mapping()
        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_agents)
        self.actions = self.actions.astype(np.int32)
    
    def _build_action_mapping(self):
        # Build the flattened action space mapping
        self.arg_counts = [max_arg + 1 for max_arg in self.max_action_args]
        self.n_actions = sum(self.arg_counts)
        self.action_map = np.zeros((self.n_actions, 2), dtype=np.int32)
        
        i = 0
        for action_type, (action_name, arg_count) in enumerate(zip(self.action_names, self.arg_counts)):
            for arg in range(arg_count):
                self.action_map[i] = (action_type, arg)
                i += 1
    
    @property
    def single_action_space(self):
        """Return flattened single action space for PufferLib."""
        if hasattr(self, 'n_actions'):
            # Create FlattenedDiscrete with original nvec for compatibility
            original_nvec = [len(self.action_names)] + self.max_action_args
            return FlattenedDiscrete(self.n_actions, original_nvec)
        return super().single_action_space
    
    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int32)
        
        # Convert flat discrete actions to MultiDiscrete format
        unflattened_actions = np.array([
            self.action_map[a] for a in actions
        ], dtype=np.int32)
        
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
