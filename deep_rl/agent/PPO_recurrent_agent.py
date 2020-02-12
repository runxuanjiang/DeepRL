#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import pdb
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance


class PPORecurrentAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        if config.network:
            self.network = config.network
        else:
            self.network = config.network_fn()
        #self.network.to(torch.device('cuda'))
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.recurrent_states = None
        self.states = self.task.reset()
        self.done = True

    def step(self):
        print("stepping")
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):

            print("rollout")

            if self.done:
                prediction, self.recurrent_states = self.network(states)
            else:
                prediction, self.recurrent_states = self.network(states, self.recurrent_states)

            self.done = False
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.add({
                'a': prediction['a'].unsqueeze(0),
                'log_pi_a': prediction['log_pi_a'],
                'ent': prediction['ent'],
                'v': prediction['v'],
            })

            dihedrals = torch.tensor(states[0][1])
            dihedrals = dihedrals.unsqueeze(0)
            states = states[0][0]
            states = states.to_data_list()
            states = states[0]
            storage.add({'edge_attr': states.edge_attr.unsqueeze(0),
                         'edge_index': states.edge_index.unsqueeze(0),
                         'pos': states.pos.unsqueeze(0),
                         'x': states.x.unsqueeze(0),
                         'dihedral': dihedrals})

            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction, self.recurrent_states = self.network(states)
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
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

        actions, log_probs_old, returns, advantages = storage.cat(['a', 'log_pi_a', 'ret', 'adv'])
        edge_attr, edge_index, pos, x, dihedral = storage.cat(['edge_attr', 'edge_index', 'pos', 'x', 'dihedral'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(edge_attr.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                sampled_edge_attr = edge_attr[batch_indices]
                sampled_edge_index = edge_index[batch_indices]
                sampled_pos = pos[batch_indices]
                sampled_x = x[batch_indices]
                sampled_dihedral = dihedral[batch_indices]

                sampled_states = []

                for i, _ in enumerate(sampled_edge_attr):
                    data = Data(
                        edge_attr = sampled_edge_attr[i],
                        edge_index = sampled_edge_index[i],
                        pos = sampled_pos[i],
                        x = sampled_x[i]
                    )
                    data = Batch.from_data_list([data])
                    sampled_states.append((data, sampled_dihedral[i].tolist()))
                
                prediction, _ = self.network(sampled_states)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()

                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()
