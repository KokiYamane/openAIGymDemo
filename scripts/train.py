import torch
from torch import nn
import sys
import os
import time
import wandb
import gym
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

sys.path.append('.')
sys.path.append('..')
from scripts.DQN import CNNQNetwork
from scripts.replay_buffer import PrioritizedReplayBuffer


class Trainer():
    def __init__(self, env, model, target_model,
                 loss_fn=nn.SmoothL1Loss(reduction='none'),
                 lr=0.001, optimizer_cls=optim.Adam):
        self.env = env
        self.optimizer = optimizer_cls(model.parameters(), lr=lr)
        self.loss_fn = loss_fn

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print('device:', self.device)

        self.model = model.to(self.device)
        self.target_model = target_model.to(self.device)

    def beta_fn(self, step, begin=0.4, end=1.0, decay=500000):
        return min(end, begin + (end - begin) * (step / decay))

    def epsilon_fn(self, step, begin=1.0, end=0.01, decay=50000):
        return max(end, begin - (begin - end) * (step / decay))

    def train_model(self, n_episodes, batch_size=32, out_dir=''):
        initial_buffer_size = 10000
        replay_buffer = PrioritizedReplayBuffer(buffer_size=100000)
        target_update_interval = 2000
        best_total_reward = 1e10
        losses = []
        total_reward_list = []
        step = 1
        for episode in range(1, n_episodes + 1):
            start = time.time()
            observation = self.env.reset()
            observation = torch.from_numpy(observation).float()
            done = False
            total_reward = 0

            while not done:
                # action
                action = self.model.act(observation, self.epsilon_fn(step))
                next_observation, reward, done, _ = self.env.step(action)
                next_observation = torch.from_numpy(next_observation).float()
                total_reward += reward

                # save experience replay buffer
                replay_buffer.push(
                    [observation, action, reward, next_observation, done])
                observation = next_observation

                # update model
                if len(replay_buffer) > initial_buffer_size:
                    loss = self.update(
                        batch_size=batch_size,
                        beta=self.beta_fn(step),
                        replay_buffer=replay_buffer,
                    )
                    losses.append(loss)

                # update target model
                if step % target_update_interval == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

                step += 1

            total_reward_list.append(total_reward)

            end = time.time()
            elapsed_time = end - start

            print('episode: {}, step: {}, reward: {}, elapsed time: {:.3f}'.format(
                episode,
                step,
                total_reward,
                elapsed_time,
            ))

            # plot loss
            plt.clf()
            plt.plot(losses)
            plt.xlabel('step')
            # plt.ylabel('MSE')
            plt.title('loss')
            plt.savefig('loss.png')

            # plot total reward
            plt.clf()
            plt.plot(total_reward_list)
            plt.xlabel('episode')
            plt.ylabel('reward')
            plt.savefig('total_reward.png')

            # save model
            if total_reward <= best_total_reward:
                best_total_reward = total_reward
                path = os.path.join(out_dir, 'model_param_best.pt')
                torch.save(self.model.state_dict(), path)
                path = os.path.join(out_dir, 'model_entire_best.pt')
                torch.save(self.model, path)

    def update(self, beta, replay_buffer, batch_size=32, gamma=0.99):
        observation, action, reward, next_observation, done, indices, weights = replay_buffer.sample(
            batch_size, beta)
        observation = observation.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_observation = next_observation.to(self.device)
        done = done.to(self.device)
        weights = weights.to(self.device)

        #　ニューラルネットワークによるQ関数の出力から, .gatherで実際に選択した行動に対応する価値を集めてきます.
        q_values = self.model(observation).gather(
            1, action.unsqueeze(1)).squeeze(1)

        # 目標値の計算なので勾配を追跡しない
        with torch.no_grad():
            # Double DQN.
            # ① 現在のQ関数でgreedyに行動を選択し,
            greedy_action_next = torch.argmax(
                self.model(next_observation), dim=1)
            # ②　対応する価値はターゲットネットワークのものを参照します.
            q_values_next = self.target_model(next_observation).gather(
                1, greedy_action_next.unsqueeze(1)).squeeze(1)

        # ベルマン方程式に基づき, 更新先の価値を計算します.
        # (1 - done)をかけているのは, ゲームが終わった後の価値は0とみなすためです.
        target_q_values = reward + gamma * q_values_next * (1 - done)

        # Prioritized Experience Replayのために, ロスに重み付けを行なって更新します.
        self.optimizer.zero_grad()
        loss = (weights * self.loss_fn(q_values, target_q_values)).mean()
        loss.backward()
        self.optimizer.step()

        #　TD誤差に基づいて, サンプルされた経験の優先度を更新します.
        replay_buffer.update_priorities(
            indices, (target_q_values - q_values).abs().detach().cpu().numpy())

        return loss.item()


def main():
    env = gym.make('CartPole-v0')
    # env = gym.make('Pendulum-v0')
    # env = ClipRewardEnv(env)
    # env = MaxAndSkipEnv(env)
    env.reset()

    model = CNNQNetwork(
        env.observation_space.shape,
        n_action=env.action_space.n)
    target_model = CNNQNetwork(
        env.observation_space.shape,
        n_action=env.action_space.n)
    train = Trainer(env, model, target_model)
    train.train_model(n_episodes=10000, batch_size=32)


if __name__ == '__main__':
    main()
