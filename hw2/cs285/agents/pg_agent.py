import copy
from cs285.infrastructure import utils
import numpy as np
from copy import deepcopy
from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer


class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # TODO: update the PG actor/policy using the given batch of data
        q_values = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, q_values, terminals)
        train_log = self.actor.update(observations, actions, advantages, q_values)
        # using helper functions to compute qvals and advantages, and
        # return the train_log obtained from updating the policy

        return train_log

    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        # TODO: return the estimated qvals based on the given rewards, using
            # either the full trajectory-based estimator or the reward-to-go estimator

        # Note: rewards_list is a list of lists of rewards with the inner list being the list of rewards for a single trajectory.
        # HINT: use the helper functions self._discounted_return and # self._discounted_cumsum (you will need to implement these).

        rewards_list_copy = copy.deepcopy(rewards_list)
        discounted_reward = [self._discounted_cumsum(self._discounted_return(rewards)) for rewards in rewards_list_copy]

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory

        # Note: q_values should first be a 2D list where the first dimension corresponds to 
        # trajectories and the second corresponds to timesteps, 
        # then flattened to a 1D numpy array.

        if not self.reward_to_go:
            q_vals_2d = [np.repeat(reward[0], len(reward)) for reward in discounted_reward]
            q_values = np.array([item for reward in q_vals_2d for item in reward])
        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            # q_values = np.flatten(np.array(rewards_list))
            q_values = np.array([item for reward in discounted_reward for item in reward])
        return q_values

    def estimate_advantage(self, obs: np.ndarray, rews_list: np.ndarray, q_values: np.ndarray, terminals: np.ndarray):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)   # 1000*1 b*1
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting error
            assert values_unnormalized.ndim == q_values.ndim
            ## TODO: values were trained with standardized q_values, so ensure
            ## that the predictions have the same mean and standard deviation as
            ## the current batch of q_values
            values = utils.normalize(values_unnormalized, np.mean(values_unnormalized), np.std(values_unnormalized))

            if self.gae_lambda is not None:
                ## append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])

                ## combine rews_list into a single array
                rews = np.concatenate(rews_list)

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    ## TODO: recursively compute advantage estimates starting from
                        ## timestep T.
                    if terminals[i]:    # 标志着一段新的trajectory
                        advantages[i+1], values[i+1] = 0, 0
                    delta_t = rews[i] + values[i] - values[i+1]
                    advantages[i] = delta_t + self.gamma * self.gae_lambda * advantages[i+1]
                    ## HINT: use terminals to handle edge cases. terminals[i]
                        ## is 1 if the state is the last in its trajectory, and
                        ## 0 otherwise.

                # remove dummy advantage
                advantages = advantages[:-1]

            else:
                ## TODO: compute advantage estimates using q_values, and values as baselines
                # pdf 中的公式（15）时序差分误差
                values = np.append(values, [0])
                rews = np.concatenate(rews_list)
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)
                for i in reversed(range(batch_size)):
                    if terminals[i]:
                        values[i+1] = 0
                    advantages[i] = rews[i] + values[i] - values[i+1]
                advantages = advantages[:-1]
        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages to have a mean of zero
        # and a standard deviation of one
        if self.standardize_advantages:
            advantages = utils.normalize(advantages, advantages.mean(), advantages.std())
        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """
        """
            combine the _discounted_return with _discounted_cumsum to form a holistic function.
        """
        return rewards

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
        G = 0
        if not self.reward_to_go:
            self.gamma = 1
        for i in reversed(range(len(rewards))):
            G = self.gamma * G + rewards[i]
            rewards[i] = G
        return rewards
