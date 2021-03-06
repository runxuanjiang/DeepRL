#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class A2CRecurrentAgentFixed(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        if config.network:
            self.network = config.network
        else:
            self.network = config.network_fn()
        self.network.to(device)
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()
        self.smh = None
        self.hidden_size = config.hidden_size
        self.h0 = torch.zeros(self.config.num_workers, self.hidden_size).to(device) #lstm hidden states
        self.c0 = torch.zeros(self.config.num_workers, self.hidden_size).to(device) #lstm cell states
        print("running A2C, tag is " + config.tag)

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            start = time.time()
            prediction, (self.h0, self.c0) = self.network(states, (self.h0, self.c0))
            self.h0 = self.h0.to(device)
            self.c0 = self.c0.to(device)
            end = time.time()

            self.logger.add_scalar('forward_pass_time', end-start, self.total_steps)

            start = time.time()
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            end = time.time()

            self.logger.add_scalar('env_step_time', end-start, self.total_steps)
            #print('step time', end-start)
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1).to(device),
                         'm': tensor(1 - terminals).unsqueeze(-1).to(device)})

            states = next_states
            #zero out lstm recurrent state if any of the environments finish
            self.h0 = self.h0.clone().detach()
            self.c0 = self.c0.clone().detach()
            for i, done in enumerate(terminals):
                if done:
                    self.h0[i] = torch.zeros(self.hidden_size)
                    self.c0[i] = torch.zeros(self.hidden_size)
            self.total_steps += config.num_workers

        self.states = states
        prediction, _ = self.network(states, (self.h0, self.c0))
        # self.smh = [s.detach() for s in self.smh]

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


        log_prob, value, returns, advantages, entropy = storage.cat(['log_pi_a', 'v', 'ret', 'adv', 'ent'])
        policy_loss = -(log_prob * advantages).mean()
        value_loss = 0.5 * (returns - value).pow(2).mean()
        entropy_loss = entropy.mean()

        self.logger.add_scalar('advantages', advantages.mean(), self.total_steps)
        self.logger.add_scalar('policy_loss', policy_loss, self.total_steps)
        self.logger.add_scalar('value_loss', value_loss, self.total_steps)
        self.logger.add_scalar('entropy_loss', entropy_loss, self.total_steps)

        start = time.time()

        self.optimizer.zero_grad()
        (policy_loss - config.entropy_weight * entropy_loss + config.value_loss_weight * value_loss).backward()
        grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
        self.logger.add_scalar('grad_norm', grad_norm, self.total_steps)
        self.optimizer.step()

        end = time.time()
        self.logger.add_scalar('backwards_pass_time', end-start, self.total_steps)
