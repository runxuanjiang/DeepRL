#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# TODO:
# - plot average rewards in matplotlib
# - look at when entropy loss is recorded


from ..network import *
from ..component import *
from .BaseAgent import *

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance

import numpy
import numpy.random

import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPORecurrentAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config #config file, contains hyperparameters and other info
        self.task = config.task_fn() #gym environment

        if config.network: #nnet used
            self.network = config.network
        else:
            self.network = config.network_fn()
        self.network.to(device)

        self.optimizer = config.optimizer_fn(self.network.parameters()) #optimization function
        self.total_steps = 0
        self.recurrent_states = None
        self.first_recurrent_states = None
        self.states = self.task.reset()
        self.recurrence = config.recurrence
        self.done = True;
        print("running PPO, tag is " + config.tag)

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length) 

        states = self.states
        self.first_recurrent_states = self.recurrent_states
        for _ in range(config.rollout_length):
            #put states and recurrent states into storage
            if self.done:
                cleared = [None for i in range(self.config.num_workers)]
                self.recurrent_states = [cleared, cleared]
                storage.add({'rs': self.recurrent_states})
            else:
                rstates = list(self.recurrent_states)
                for i, state in enumerate(rstates):
                    rstates[i] = state.detach()
                storage.add({'rs': rstates})

            #run the neural net once to get prediction
            start = time.time()
            if self.done:
                prediction, self.recurrent_states = self.network(states)
            else:
                prediction, self.recurrent_states = self.network(states, self.recurrent_states)
            end = time.time()
            self.logger.add_scalar('forward_pass_time', end-start, self.total_steps)

            self.done = False

            #step the environment with the action determined by the prediction
            start = time.time()
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            end = time.time()
            self.logger.add_scalar('env_step_time', end-start, self.total_steps)

            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)

            #add everything to storage
            storage.add(prediction)
            storage.add({'s': states})
            storage.add({'r': tensor(rewards).unsqueeze(-1).to(device),
                         'm': tensor(1 - terminals).unsqueeze(-1).to(device)})
            states = next_states

            self.total_steps += config.num_workers

        self.states = states

        prediction, self.recurrent_states = self.network(states)

        #TODO:This could possibly be an issue, would this prediction ever be used?
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1))).to(device)
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        log_probs_old, values, returns, advantages= storage.cat(['log_pi_a', 'v', 'ret', 'adv'])
        log_probs_old = log_probs_old.detach()
        values = values.detach()
        states = storage.s
        rc_states = storage.rs
        
        advantages = (advantages - advantages.mean()) / advantages.std()

        self.logger.add_scalar('advantages', advantages.mean(), self.total_steps)


        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(len(states)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()

                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                sampled_rc_states = [rc_states[i] for i in batch_indices]
                sampled_states = []
                for i in batch_indices:
                    sampled_states = sampled_states + list(states[i])

                prediction, _ = self.network(sampled_states, sampled_rc_states)

                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()

                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.optimizer.step()

                self.logger.add_scalar('entropy_loss', prediction['ent'].mean(), self.total_steps)
                self.logger.add_scalar('policy_loss', policy_loss, self.total_steps)
                self.logger.add_scalar('value_loss', value_loss, self.total_steps)


        #Training Loop for self.recursive
        # for _ in range(config.optimization_epochs):
        #     indices = numpy.arange(0, self.config.rollout_length, self.recurrence)
        #     indices = numpy.random.permutation(indices)
        #     batch_size = config.mini_batch_size // self.recurrence
        #     starting_batch_indices = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
        #     for starting_indices in starting_batch_indices:
        #         datapoints = 0
        #         batch_entropy = 0
        #         batch_value_loss = 0
        #         batch_policy_loss = 0
        #         batch_loss = 0
        #         for index in starting_indices:
        #             recursive_state = rc_states[index]
        #             for i in range(index, index+self.recurrence):

        #                 prediction, recursive_state = self.network(states[i], recursive_state)

        #                 entropy = prediction['ent'].mean()

        #                 ratio = (prediction['log_pi_a'] - log_probs_old[i]).exp()

        #                 obj = ratio * advantages[i]
        #                 obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
        #                                         1.0 + self.config.ppo_ratio_clip) * advantages[i]


        #                 policy_loss = -torch.min(obj, obj_clipped).mean()

        #                 value_clipped = value[i] + torch.clamp(prediction['v'] - value[i], -self.config.ppo_ratio_clip, self.config.ppo_ratio_clip)
        #                 surr1 = (prediction['v'] - returns[i]).pow(2)
        #                 surr2 = (value_clipped - returns[i]).pow(2)

        #                 value_loss = torch.max(surr1, surr2).mean()



        #                 loss = policy_loss - (config.entropy_weight * entropy) + config.value_loss_weight * value_loss

        #                 batch_entropy += entropy.item()
        #                 batch_policy_loss += policy_loss.item()
        #                 batch_value_loss += value_loss.item()
        #                 batch_loss += loss;
        #                 datapoints += 1


        #         batch_entropy /= datapoints
        #         batch_policy_loss /= datapoints
        #         batch_value_loss /= datapoints

        #         self.logger.add_scalar('entropy_loss', batch_entropy, self.total_steps)
        #         self.logger.add_scalar('policy_loss', batch_policy_loss, self.total_steps)
        #         self.logger.add_scalar('value_loss', batch_value_loss, self.total_steps)

        #         self.optimizer.zero_grad()
        #         batch_loss.backward()
        #         nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        #         self.optimizer.step()
        

            
