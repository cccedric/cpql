
import copy
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from agents.karras_diffusion import KarrasDenoiser
from agents.model import MLP, Critic
from agents.helpers import EMA
from agents.replay_memory import ReplayMemory


class CPQL(object):
    def __init__(self,
                 device,
                 state_dim,
                 action_dim,
                 rl_type="offline",
                 action_space=None,
                 discount=0.99,
                 max_q_backup=False,
                 alpha=1.0,
                 eta=1.0,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 q_mode='q',
                 sigma_max=80.0,
                 expectile=0.6,
                 sampler="onestep",
                 memory_size=1e6,
                 ):

        self.actor = MLP(state_dim=state_dim, action_dim=action_dim, device=device).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.diffusion = KarrasDenoiser(action_dim=action_dim, 
                                        sigma_max=sigma_max,
                                        device=device,
                                        sampler=sampler,)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = (action_space.high - action_space.low) / 2.
            self.action_bias = (action_space.high + action_space.low) / 2.

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim, rl_type).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.rl_type = rl_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.alpha = alpha  # bc weight
        self.eta = eta  # q_learning weight
        self.expectile = expectile
        self.device = device
        self.max_q_backup = max_q_backup
        self.q_mode = q_mode
        
        self.memory = ReplayMemory(state_dim, action_dim, memory_size, device)

    def append_memory(self, state, action, reward, next_state, not_done):
        action = (action - self.action_bias) / self.action_scale
        self.memory.append(state, action, reward, next_state, not_done)

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.actor_target, self.actor)
        self.ema.update_model_average(self.critic_target, self.critic)

    def train(self, replay_buffer, batch_size=100, log_writer=None):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        """ Q Training """
        current_q1, current_q2 = self.critic(state, action)
        if self.q_mode == 'q':
            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.diffusion.sample(model=self.actor, state=next_state_rpt)
                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.diffusion.sample(model=self.actor, state=next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
            target_q = (reward + not_done * self.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q) 
        elif self.q_mode == 'q_v':
            def expectile_loss(diff, expectile=0.8):
                weight = torch.where(diff > 0, expectile, (1 - expectile))
                return weight * (diff**2)
            
            with torch.no_grad():
                q = self.critic.q_min(state, action)
            v = self.critic.v(state)
            value_loss = expectile_loss(q - v, self.expectile).mean()

            current_q1, current_q2 = self.critic(state, action)
            with torch.no_grad():
                next_v = self.critic.v(next_state)
            target_q = (reward + not_done * self.discount * next_v).detach()
        
            critic_loss = value_loss + F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q) 

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        """ Policy Training """
        compute_bc_losses = functools.partial(self.diffusion.consistency_losses,
                                              model=self.actor,
                                              x_start=action,
                                              num_scales=40,
                                              target_model=self.actor_target,
                                              state=state,)

        bc_losses = compute_bc_losses()
        bc_loss = bc_losses["loss"].mean()
        consistency_loss = bc_losses["consistency_loss"].mean()
        recon_loss = bc_losses["recon_loss"].mean()

        new_action = self.diffusion.sample(model=self.actor, state=state)

        q1_new_action, q2_new_action = self.critic(state, new_action)
        if self.rl_type == "offline":
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        else:
            q_loss = - torch.min(q1_new_action, q2_new_action).mean()
        
        actor_loss = self.alpha * bc_loss + self.eta * q_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0: 
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.actor_optimizer.step()

        """ Step Target network """
        if self.step % self.update_ema_every == 0:
            self.step_ema()

        self.step += 1

        metric['actor_loss'].append(actor_loss.item())
        metric['bc_loss'].append(bc_loss.item())
        metric['ql_loss'].append(q_loss.item())
        metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state, num=10):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.diffusion.sample(model=self.actor, state=state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
        
        idx = torch.multinomial(F.softmax(q_value), 1)
        action = action[idx].cpu().data.numpy().flatten()

        action = action.clip(-1, 1)
        action = action * self.action_scale + self.action_bias
        return action

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))


