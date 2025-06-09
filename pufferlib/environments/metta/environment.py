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

class MettaPuff(MettaGridEnv):
    def __init__(self, config, render_mode='human', buf=None, seed=0):
        super().__init__(config, render_mode=render_mode, buf=buf)
        self.action_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_agents)
        self.actions = self.actions.astype(np.int32)

    @property
    def single_action_space(self):
        # Return a consistent action space of MultiDiscrete([8, 4])
        # This allows the policy to generate any action type (0-7) with any argument (0-3)
        # We'll clamp invalid arguments in the step method
        return gymnasium.spaces.MultiDiscrete([8, 4], dtype=np.int32)

    def _clamp_action_arguments(self, actions):
        """
        Clamp action arguments to valid ranges for each action type.
        
        Args:
            actions: numpy array of shape (num_agents, 2) where [:,0] is action type and [:,1] is argument
            
        Returns:
            clamped_actions: numpy array with invalid arguments clamped to valid ranges
        """
        # Create a copy to avoid modifying the original
        clamped_actions = actions.copy()
        
        # Get action names to map indices to action types
        action_names = self.action_names
        
        # Define valid argument ranges for each action type
        # Based on the max_arg() values from the C++ action handlers
        action_arg_limits = {
            'noop': 0,              # max_arg() = 0, accepts [0]
            'move': 1,              # max_arg() = 1, accepts [0, 1] 
            'rotate': 3,            # max_arg() = 3, accepts [0, 1, 2, 3]
            'get_output': 0,        # max_arg() = 0, accepts [0]
            'put_recipe_items': 0,  # max_arg() = 0, accepts [0]
            'attack': 0,            # max_arg() = 0, accepts [0]
            'attack_nearest': 0,    # max_arg() = 0, accepts [0]
            'swap': 0,              # max_arg() = 0, accepts [0]
            'change_color': 3,      # max_arg() = 3, accepts [0, 1, 2, 3]
        }
        
        # Clamp arguments for each agent's action
        for i in range(clamped_actions.shape[0]):
            action_type = clamped_actions[i, 0]
            action_arg = clamped_actions[i, 1]
            
            # Get the action name for this action type index
            if 0 <= action_type < len(action_names):
                action_name = action_names[action_type]
                max_valid_arg = action_arg_limits.get(action_name, 0)
                
                # Clamp the argument to the valid range [0, max_valid_arg]
                clamped_actions[i, 1] = np.clip(action_arg, 0, max_valid_arg)
        
        return clamped_actions

    def step(self, actions):
        # Clamp action arguments to valid ranges before passing to the environment
        clamped_actions = self._clamp_action_arguments(actions)
        
        # Call parent step with clamped actions
        obs, rew, term, trunc, info = super().step(clamped_actions)

        if all(term) or all(trunc):
            self.reset()
            # Remove all keys starting with 'agent_raw'
            for k in list(info.keys()):
                if k.startswith('agent_raw'):
                    del info[k]
            if 'episode_rewards' in info:
                info['score'] = info['episode_rewards']

        else:
            info = []

        return obs, rew, term, trunc, [info]
