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


class PPORecurrentAgentRecurrence(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config #config file, contains hyperparameters and other info
        self.task = config.task_fn() #gym environment wrapper
        self.hidden_size = config.hidden_size

        if config.network: #nnet used
            self.network = config.network
        else:
            self.network = config.network_fn()
        self.network.to(device)

        self.optimizer = config.optimizer_fn(self.network.parameters()) #optimization function
        self.total_steps = 0
        self.states = self.task.reset()
        self.h0 = torch.zeros(self.config.num_workers, self.hidden_size).to(device) #lstm hidden states
        self.c0 = torch.zeros(self.config.num_workers, self.hidden_size).to(device) #lstm cell states
        self.recurrence = self.config.recurrence
        print("running PPO, tag is " + config.tag)

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length) 

        states = self.states

        ##############################################################################################
        #Sampling Loop
        ##############################################################################################
        for _ in range(config.rollout_length):

            #add recurrent states (lstm hidden and lstm cell states) to storage
            storage.add({
                'h0' : self.h0.to(device),
                'c0' : self.c0.to(device)
            })

            #run the neural net once to get prediction
            prediction, (self.h0, self.c0) = self.network(states, (self.h0, self.c0))
            self.h0 = self.h0.to(device)
            self.c0 = self.c0.to(device)

            #step the environment with the action determined by the prediction
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)

            #add everything to storage
            storage.add(prediction)
            storage.add({
                's' : tensor(states).to(device),
                'r': tensor(rewards).unsqueeze(-1).to(device),
                'm': tensor(1 - terminals).unsqueeze(-1).to(device)
                })
            states = next_states
            
            #zero out lstm recurrent state if any of the environments finish
            for i, done in enumerate(terminals):
                if done:
                    self.h0[i] = torch.zeros(self.hidden_size)
                    self.c0[i] = torch.zeros(self.hidden_size)

            self.total_steps += config.num_workers

        self.states = states

        prediction, _ = self.network(states, (self.h0, self.c0))

        storage.add(prediction)
        storage.placeholder()


        #############################################################################################
        #Calculate advantages and returns and set up for training
        #############################################################################################

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

        storage.a = storage.a[:self.config.rollout_length]
        storage.log_pi_a = storage.log_pi_a[:self.config.rollout_length]
        storage.v = storage.v[:self.config.rollout_length]


        actions = torch.stack(storage.a, 1).reshape(-1)
        log_probs_old = torch.cat(storage.log_pi_a, 1).reshape(-1)
        values = torch.cat(storage.v, 1).reshape(-1)
        returns = torch.cat(storage.ret, 1).reshape(-1)
        advantages = torch.cat(storage.adv, 1).reshape(-1)

        log_probs_old = log_probs_old.detach()
        values = values.detach()
        states = torch.stack(storage.s, 1).view(-1, 4)
        h0 = torch.stack(storage.h0, 1).view(-1, self.hidden_size)
        c0 = torch.stack(storage.c0, 1).view(-1, self.hidden_size)

        
        advantages = (advantages - advantages.mean()) / advantages.std()

        self.logger.add_scalar('advantages', advantages.mean(), self.total_steps)



        ############################################################################################
        #Training Loop
        ############################################################################################
        for _ in range(config.optimization_epochs):
            indices = numpy.arange(0, self.config.rollout_length * self.config.num_workers, self.recurrence);
            indices = numpy.random.permutation(indices);
            num_indices = config.mini_batch_size // self.recurrence
            starting_batch_indices = [indices[i:i+num_indices] for i in range(0, len(indices), num_indices)]
            for starting_indices in starting_batch_indices:
                batch_entropy = 0
                batch_value_loss = 0
                batch_policy_loss = 0
                batch_loss = 0

                sampled_h0 = h0[starting_indices]
                sampled_c0 = c0[starting_indices]

                for i in range(self.recurrence):
                    sampled_actions = actions[starting_indices + i]
                    sampled_log_probs_old = log_probs_old[starting_indices + i]
                    sampled_values = values[starting_indices + i]
                    sampled_returns = returns[starting_indices + i]
                    sampled_advantages = advantages[starting_indices + i]
                    sampled_states = states[starting_indices + i]

                    prediction, (sampled_h0, sampled_c0) = self.network(sampled_states, (sampled_h0, sampled_c0), sampled_actions)

                    entropy = prediction['ent'].mean()
                    
                    prediction['log_pi_a'] = prediction['log_pi_a'].reshape(-1)
                    prediction['v'] = prediction['v'].reshape(-1)

                    ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                    obj = ratio * sampled_advantages
                    obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                            1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                    policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()

                    value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                    loss = policy_loss + value_loss

                    batch_entropy += entropy.item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss;


            batch_entropy /= self.recurrence
            batch_policy_loss /= self.recurrence
            batch_value_loss /= self.recurrence
            batch_loss /= self.recurrence

            self.logger.add_scalar('entropy_loss', batch_entropy, self.total_steps)
            self.logger.add_scalar('policy_loss', batch_policy_loss, self.total_steps)
            self.logger.add_scalar('value_loss', batch_value_loss, self.total_steps)

            self.optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
            self.optimizer.step()


            
