import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        ##  TODO , qa_value 是ob-> action的映射吗？
        actions =  self.critic.qa_values(observation)
        action = actions.argmax()
        return action.squeeze()