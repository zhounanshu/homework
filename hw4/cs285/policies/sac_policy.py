from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: get this from previous HW
        entropy = self.log_alpha.exp()
        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: get this from previous HW\
        obs = ptu.from_numpy(obs)
        if sample:
            action = self(obs).sample()
        else:
            action = self(obs).mean()
        # clip action
        action = action.clip(action, *self.action_range)
        action = ptu.to_numpy(action)
        return action

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: get this from previous HW
        mu = self.mean_net(observation)
        logstd = sac_utils.TanhTransform()(self.logstd)
        logstd_min, logstd_max = self.log_std_bounds
        logstd = torch.clip(logstd, logstd_min, logstd_max)
        std = logstd.exp()
        action_distribution = sac_utils.SquashedNormal(mu, std)
        return action_distribution

    def update(self, obs, critic):
        # TODO: get this from previous HW
        ac_dist = self(ptu.from_numpy(obs))
        action = self.ac_dist.rsample() # ??
        log_prob = ac_dist.log_prob(action)
        q1_value, q2_value = critic(obs, action)
        actor_loss =  (self.alpha * log_prob.detach() - torch.min(q1_value,  q2_value)).mean()
        
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        alpha_loss =  (-self.alpha * log_prob - self.target_entropy).detach().mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return actor_loss, alpha_loss, self.alpha