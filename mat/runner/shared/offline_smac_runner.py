import time
import wandb
import numpy as np
from functools import reduce
import torch
from mat.runner.shared.offline_base_runner import OfflineRunner
from torch.utils.data.dataloader import DataLoader
from mat.utils.util import get_gard_norm

def _t2n(x):
    return x.detach().cpu().numpy()

class OfflineSMACRunner(OfflineRunner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(OfflineSMACRunner, self).__init__(config)

    def run(self):
        loader = DataLoader(self.buffer, shuffle=True, pin_memory=True, drop_last=True,
                            batch_size=self.all_args.batch_size,
                            num_workers=16)
        for epoch in range(self.train_epoch):
            train_info = {}
            train_info['policy_loss'] = 0
            train_info['grad_norm'] = 0

            for mini_batch_iter, (obss, actions, available_actions) in enumerate(loader):
                action_log_probs, _, _ = self.policy.transformer(None, obss, actions, available_actions)
                loss = (-action_log_probs).mean()  # cross entropy loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                if self.all_args.use_max_grad_norm:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.transformer.parameters(),
                                                         self.all_args.max_grad_norm)
                else:
                    grad_norm = get_gard_norm(self.policy.transformer.parameters())
                self.policy.optimizer.step()

                train_info['policy_loss'] += loss.mean().item()
                train_info['grad_norm'] += grad_norm.mean().item()

            if epoch % self.log_interval == 0:
                print("epoch: {}, loss: {}, grad norm: {}."
                      .format(epoch, train_info['policy_loss'], train_info['grad_norm']))
                self.log_train(train_info, epoch)

            # save model
            if epoch % self.save_interval == 0 or epoch == self.train_epoch - 1:
                self.save(epoch)

            # eval
            if epoch % self.eval_interval == 0 and self.use_eval:
                self.eval(epoch)

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:

            self.policy.eval()
            eval_actions, eval_rnn_states = \
                self.policy.act(np.concatenate(eval_share_obs),
                                np.concatenate(eval_obs),
                                np.concatenate(eval_rnn_states),
                                np.concatenate(eval_masks),
                                np.concatenate(eval_available_actions),
                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                # self.eval_envs.save_replay()
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
