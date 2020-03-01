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
        self.states = self.task.reset()
        self.done = True

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length) 
        #rollout_length should be the same for mini_batch_size. 
        #This ensures that the recurrent_states match up with the proper corresponding input for the neural net.
        states = self.states
        for _ in range(config.rollout_length):

            start = time.time()
            #TODO: Currently, environment still runs with recurrent states after resetting,
            #Need to make sure the environment doesn't run with reccurent states directly after resetting
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

        prediction, self.recurrent_states = self.network(states)
        #We don't need to add the prediction to storage here
        # storage.add(prediction)

        # storage.add({
        #     'a': prediction['a'].unsqueeze(0),
        #     'log_pi_a': prediction['log_pi_a'],
        #     'ent': prediction['ent'],
        #     'v': prediction['v'],
        # })
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

        log_probs_old, returns, advantages, entropy= storage.cat(['log_pi_a', 'ret', 'adv', 'ent'])
        states = storage.s
        
        pdb.set_trace()
        log_probs_old = log_probs_old.detach()
        #advantages = (advantages - advantages.mean()) / advantages.std()

        entropy_loss = entropy.mean()

        self.logger.add_scalar('advantages', advantages.mean(), self.total_steps)

        for _ in range(config.optimization_epochs):
            for _ in range(config.rollout_length):
            # sampler = random_sample(np.arange(edge_attr.size(0)), config.mini_batch_size)
            # for batch_indices in sampler:
                # batch_indices = tensor(batch_indices).long()
                # sampled_log_probs_old = log_probs_old[batch_indices]
                # sampled_returns = returns[batch_indices]
                # sampled_advantages = advantages[batch_indices]

                # sampled_edge_attr = edge_attr[batch_indices]
                # sampled_edge_index = edge_index[batch_indices]
                # sampled_pos = pos[batch_indices]
                # sampled_x = x[batch_indices]
                # sampled_dihedral = dihedral[batch_indices]

                # sampled_states = []

                # for i, _ in enumerate(sampled_edge_attr):
                #     data = Data(
                #         edge_attr = sampled_edge_attr[i],
                #         edge_index = sampled_edge_index[i],
                #         pos = sampled_pos[i],
                #         x = sampled_x[i]
                #     )
                #     data = Batch.from_data_list([data])
                #     sampled_states.append((data, sampled_dihedral[i].tolist()))

                
                prediction, _ = self.network(sampled_states)

                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()

                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages


                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()


                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()


                self.logger.add_scalar('entropy_loss', prediction['ent'].mean(), self.total_steps)
                self.logger.add_scalar('policy_loss', policy_loss, self.total_steps)
                self.logger.add_scalar('value_loss', value_loss, self.total_steps)


                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()
