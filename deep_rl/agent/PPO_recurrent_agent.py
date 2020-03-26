#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################



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

        self.opt = config.optimizer_fn(self.network.parameters()) #optimization function
        self.total_steps = 0
        self.recurrent_states = None
        self.first_recurrent_states = None
        self.states = self.task.reset()
        self.recurrence = config.recurrence
        self.done = True

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length) 

        states = self.states
        self.first_recurrent_states = self.recurrent_states
        for _ in range(config.rollout_length):

            #because nontype doesn't have .detach()
            if not self.done:
                rstates = list(self.recurrent_states)
                for i, state in enumerate(rstates):
                    rstates[i] = state.detach()
                
                storage.add({'rs': rstates})
            else:
                storage.add({'rs': self.recurrent_states})

            start = time.time()
            #TODO: Currently, environment still runs with recurrent states after resetting,
            #Need to make sure the environment doesn't run with reccurent states directly after resetting
            with torch.no_grad():
                if self.done:
                    print("Environment Just Reset, Running NN Without Recurrent States")
                    prediction, self.recurrent_states = self.network(states)
                else:
                    print("Running NN With Recurrent States")
                    prediction, self.recurrent_states = self.network(states, self.recurrent_states)

            end = time.time()

            self.logger.add_scalar('forward_pass_time', end-start, self.total_steps)

            self.done = False

            start = time.time()
            #One environment interaction
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            end = time.time()
            self.logger.add_scalar('env_step_time', end-start, self.total_steps)

            #self.done = terminals[0]

            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.add(prediction)
            #Use this format if the "a" or action parameter is used.
            
            # storage.add({
            #     'a': prediction['a'].unsqueeze(0),
            #     'log_pi_a': prediction['log_pi_a'],
            #     'ent': prediction['ent'],
            #     'v': prediction['v'],
            # })
            

            # dihedrals = torch.tensor(states[0][1])
            # dihedrals = dihedrals.unsqueeze(0)
            # states = states[0][0]
            # states = states.to_data_list()
            # states = states[0]
            # storage.add({'edge_attr': states.edge_attr.unsqueeze(0).to(device),
            #              'edge_index': states.edge_index.unsqueeze(0).to(device),
            #              'pos': states.pos.unsqueeze(0).to(device),
            #              'x': states.x.unsqueeze(0).to(device),
            #              'dihedral': dihedrals})
            storage.add({'s': states})
            storage.add({'r': tensor(rewards).unsqueeze(-1).to(device),
                         'm': tensor(1 - terminals).unsqueeze(-1).to(device)})
            """if (self.done):
                states = self.task.reset()
            else:
                states = next_states"""
            states = next_states

            self.total_steps += config.num_workers

        self.states = states

        with torch.no_grad():
            prediction, self.recurrent_states = self.network(states)

        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1))).to(device)
        returns = prediction['v']
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i]
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages
            storage.ret[i] = returns

        log_probs_old, returns, advantages, entropy= storage.cat(['log_pi_a', 'ret', 'adv', 'ent'])
        states = storage.s
        rc_states = storage.rs
        
        log_probs_old = log_probs_old
        advantages = (advantages - advantages.mean()) / advantages.std()

        self.logger.add_scalar('advantages', advantages.mean(), self.total_steps)


        for _ in range(config.optimization_epochs):
            indices = numpy.arange(0, self.config.rollout_length, self.recurrence)
            indices = numpy.random.permutation(indices)
            batch_size = config.mini_batch_size // self.recurrence
            starting_batch_indices = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
            for starting_indices in starting_batch_indices:
                datapoints = 0
                batch_entropy = 0
                batch_value_loss = 0
                batch_policy_loss = 0
                batch_loss = 0
                for index in starting_indices:
                    recursive_state = rc_states[index]
                    for i in range(index, index+self.recurrence):

                        prediction, recursive_state = self.network(states[i], recursive_state)

                        entropy = prediction['ent'].mean()

                        ratio = (prediction['log_pi_a'] - log_probs_old[i]).exp()

                        obj = ratio * advantages[i]
                        obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                                1.0 + self.config.ppo_ratio_clip) * advantages[i]


                        policy_loss = -torch.min(obj, obj_clipped).mean()# - config.entropy_weight * prediction['ent']


                        value_loss = 0.5 * (returns[i] - prediction['v']).pow(2)

                        loss = policy_loss - (config.entropy_weight * entropy) + config.value_loss_weight * value_loss

                        batch_entropy += entropy.item()
                        batch_policy_loss += policy_loss.item()
                        batch_value_loss += value_loss.item()
                        batch_loss += loss
                        datapoints += 1


                batch_entropy /= datapoints
                batch_policy_loss /= datapoints
                batch_value_loss /= datapoints
                batch_loss /= datapoints

                self.logger.add_scalar('entropy_loss', batch_entropy, self.total_steps)
                self.logger.add_scalar('policy_loss', batch_policy_loss, self.total_steps)
                self.logger.add_scalar('value_loss', batch_value_loss, self.total_steps)

                self.opt.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()
        

            
