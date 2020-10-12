import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.base import LSTM_Base

class Stochastic_Actor:
    """
    The base class for stochastic actors.
    """
    def __init__(self, latent, action_dim, env_name, bounded, fixed_std=None):

        self.action_dim        = action_dim
        self.env_name          = env_name
        self.means             = nn.Linear(latent, action_dim)
        self.bounded           = bounded

        if fixed_std is None:
            self.log_stds = nn.Linear(latent, action_dim)
        self.fixed_std = fixed_std

    def _get_dist_params(self, state, update=False):
        state = self.normalize_state(state, update=update)
        x = self._base_forward(state)

        mu = self.means(x)

        if self.fixed_std is None:
            std = torch.clamp(self.log_stds(x), -2, 1).exp()
        else:
            std = self.fixed_std

        return mu, std

    def stochastic_forward(self, state, deterministic=True, update=False, log_probs=False):
        mu, sd = self._get_dist_params(state, update=update)

        if not deterministic or log_probs:
            dist = torch.distributions.Normal(mu, sd)
            sample = dist.rsample()

        if self.bounded:
            action = torch.tanh(mu) if deterministic else torch.tanh(sample)
        else:
            action = mu if deterministic else sample

        if log_probs:
            log_prob = dist.log_prob(sample)
            if self.bounded:
                log_prob -= torch.log((1 - torch.tanh(sample).pow(2)) + 1e-6)

            return action, log_prob.sum(1, keepdim=True)
        else:
            return action

    def pdf(self, state):
        mu, sd = self._get_dist_params(state)
        return torch.distributions.Normal(mu, sd)


class LSTM_Stochastic_Actor(LSTM_Base, Stochastic_Actor):
    """
    A class inheriting from LSTM_Base and Stochastic_Actor
    which implements a recurrent stochastic policy.
    """
    def __init__(self, input_dim, action_dim, layers=(128, 128), env_name=None, bounded=False, fixed_std=None):

        LSTM_Base.__init__(self, input_dim, layers)
        Stochastic_Actor.__init__(self, layers[-1], action_dim, env_name, bounded, fixed_std=fixed_std)

        self.is_recurrent = True
        self.init_hidden_state()

    def forward(self, x, deterministic=True, update_norm=False, return_log_probs=False):
        return self.stochastic_forward(x, deterministic=deterministic, update=update_norm, log_probs=return_log_probs)
