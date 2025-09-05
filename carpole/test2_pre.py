import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 设置 Matplotlib 字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# 解决负号显示问题
plt.rcParams["axes.unicode_minus"] = False
# 定义经验回放缓冲区
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# 定义DQN神经网络（输入状态，输出两个动作的Q值）
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # 输入层→隐藏层
        self.fc2 = nn.Linear(64, 64)  # 隐藏层→隐藏层
        self.fc3 = nn.Linear(64, action_size)  # 隐藏层→输出层（动作Q值）

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # 输出每个动作的Q值


# DQN智能体
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # 超参数
        self.gamma = 0.99  # 折扣因子（未来奖励的衰减）
        self.epsilon = 1.0  # ε-贪婪策略（初始100%随机探索）
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减系数
        self.learning_rate = 0.001  # 学习率
        self.batch_size = 64  # 经验回放批次大小
        self.memory = ReplayBuffer(10000)  # 经验回放缓冲区容量

        # 主网络（当前策略）和目标网络（稳定目标）
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 初始参数同步
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def select_action(self, state):
        # ε-贪婪策略：以ε概率随机选动作，否则选Q值最大的动作
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)  # 随机动作
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state)
                q_values = self.policy_net(state)
                return q_values.max(0)[1].item()  # Q值最大的动作

    def learn(self):
        # 经验不足时不学习
        if len(self.memory) < self.batch_size:
            return

        # 从经验回放中采样
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # 转换为张量
        states = torch.FloatTensor(batch.state)
        actions = torch.LongTensor(batch.action).unsqueeze(1)
        next_states = torch.FloatTensor(batch.next_state)
        rewards = torch.FloatTensor(batch.reward)
        dones = torch.FloatTensor(batch.done)

        # 计算当前Q值（主网络）和目标Q值（目标网络）
        current_q = self.policy_net(states).gather(1, actions)  # 每个状态-动作对的Q值
        next_q = self.target_net(next_states).max(1)[0].detach()  # 下一状态的最大Q值
        target_q = rewards + (1 - dones) * self.gamma * next_q  # 目标Q值（贝尔曼方程）

        # 计算损失（MSE）并更新主网络
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self):
        # 同步目标网络参数（定期更新，提高稳定性）
        self.target_net.load_state_dict(self.policy_net.state_dict())


# 训练函数
def train_agent(episodes=200, target_score=195):
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]  # 4
    action_size = env.action_space.n  # 2
    agent = DQNAgent(state_size, action_size)

    # 记录训练指标
    scores = []  # 每回合的奖励（得分）
    epsilons = []  # 每回合的探索率
    steps_per_episode = []  # 每回合的步数
    converged = False
    convergence_episode = None

    for e in range(episodes):
        state, _ = env.reset()
        score = 0  # 本回合得分
        steps = 0  # 本回合步数

        while True:
            # 选动作→执行→存经验→学习
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # 回合结束标志

            # 存储经验
            agent.memory.push(state, action, next_state, reward, done)
            # 学习（更新主网络）
            agent.learn()

            state = next_state
            score += reward
            steps += 1

            if done:
                # 每10回合更新一次目标网络
                if e % 10 == 0:
                    agent.update_target_net()
                print(f"回合 {e + 1}/{episodes}, 得分: {score}, 步数: {steps}, ε: {agent.epsilon:.2f}")
                scores.append(score)
                steps_per_episode.append(steps)
                epsilons.append(agent.epsilon)

                # 检查是否收敛（连续10回合得分超过目标值）
                if not converged and e >= 9:
                    recent_scores = scores[-10:]
                    if np.mean(recent_scores) >= target_score:
                        converged = True
                        convergence_episode = e + 1  # 记录收敛回合数
                        print(f"在第 {convergence_episode} 回合达到收敛标准！")

                break

    env.close()
    return scores, steps_per_episode, epsilons, convergence_episode, agent


# 计算移动平均值
def moving_average(values, window_size=10):
    """计算移动平均值，用于平滑曲线展示收敛趋势"""
    if len(values) < window_size:
        return [np.mean(values[:i + 1]) for i in range(len(values))]
    return [np.mean(values[i:i + window_size]) for i in range(len(values) - window_size + 1)]


# 展示训练成果
def plot_metrics(scores, steps, epsilons, convergence_episode):
    """可视化训练过程中的各项指标"""
    window_size = 10
    avg_scores = moving_average(scores, window_size)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DQN智能体训练性能指标', fontsize=16)

    # 1. 每回合得分
    axs[0, 0].plot(scores, label='每回合得分', alpha=0.5)
    axs[0, 0].plot(range(window_size - 1, len(scores)), avg_scores, 'r-', label=f'{window_size}回合移动平均')
    if convergence_episode:
        axs[0, 0].axvline(x=convergence_episode - 1, color='g', linestyle='--',
                          label=f'收敛点: 第{convergence_episode}回合')
    axs[0, 0].set_xlabel('回合数')
    axs[0, 0].set_ylabel('得分')
    axs[0, 0].set_title('每回合得分及移动平均值')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 2. 移动平均得分（单独放大展示）
    axs[0, 1].plot(range(window_size - 1, len(scores)), avg_scores, 'r-', label=f'{window_size}回合移动平均')
    if convergence_episode:
        axs[0, 1].axvline(x=convergence_episode - 1, color='g', linestyle='--',
                          label=f'收敛点: 第{convergence_episode}回合')
    axs[0, 1].set_xlabel('回合数')
    axs[0, 1].set_ylabel('平均得分')
    axs[0, 1].set_title('移动平均得分（收敛趋势）')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 3. 每回合步数
    axs[1, 0].plot(steps, label='每回合步数')
    axs[1, 0].set_xlabel('回合数')
    axs[1, 0].set_ylabel('步数')
    axs[1, 0].set_title('每回合持续步数')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    axs[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 4. 探索率变化
    axs[1, 1].plot(epsilons, label='ε值', color='orange')
    axs[1, 1].set_xlabel('回合数')
    axs[1, 1].set_ylabel('探索率 (ε)')
    axs[1, 1].set_title('探索率衰减曲线')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    axs[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # 打印关键统计信息
    print("\n===== 训练成果统计 =====")
    print(f"总训练回合数: {len(scores)}")
    print(f"最高得分: {max(scores)}")
    print(f"最低得分: {min(scores)}")
    print(f"最后10回合平均得分: {np.mean(scores[-10:]):.2f}")
    print(f"总平均得分: {np.mean(scores):.2f}")
    if convergence_episode:
        print(f"达到收敛的回合数: {convergence_episode}")
    else:
        print("在训练过程中未达到收敛标准")


# 训练并可视化结果
if __name__ == "__main__":
    # 训练智能体（200回合）
    episodes = 200
    target_score = 195  # CartPole-v1的解决标准是连续100回合平均得分≥475，这里降低标准用于演示
    scores, steps, epsilons, convergence_episode, agent = train_agent(episodes, target_score)

    # 展示训练指标
    plot_metrics(scores, steps, epsilons, convergence_episode)

    # 测试训练好的智能体（可视化）
    env = gym.make("CartPole-v1", render_mode="human")
    state, _ = env.reset()
    score = 0
    steps = 0
    while True:
        action = agent.select_action(state)  # 此时ε已很小，基本用最优策略
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        score += reward
        steps += 1
        if terminated or truncated:
            print(f"\n测试结果 - 得分: {score}, 步数: {steps}")
            break
    env.close()
