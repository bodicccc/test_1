import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from scipy import stats

# 设置 Matplotlib 字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# 解决负号显示问题
plt.rcParams["axes.unicode_minus"] = False
# 经验回放缓冲区
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


# 改进的DQN网络（增加网络容量）
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # 增加节点数
        self.fc2 = nn.Linear(128, 128)  # 增加一层隐藏层
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)  # 输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# 优化后的DQN智能体
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # 优化超参数
        self.gamma = 0.995  # 稍微提高折扣因子，更重视未来奖励
        self.epsilon = 1.0
        self.epsilon_min = 0.001  # 更低的最小探索率
        self.epsilon_decay = 0.997  # 更慢的衰减，保证充分探索
        self.learning_rate = 0.0005  # 更小的学习率，避免震荡
        self.batch_size = 128  # 更大的批次
        self.memory = ReplayBuffer(50000)  # 更大的经验缓冲区

        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.target_update_freq = 5  # 更频繁地更新目标网络（每5回合）

        # 新增：记录训练过程中的损失
        self.loss_history = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state)
                q_values = self.policy_net(state)
                return q_values.max(0)[1].item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(batch.state)
        actions = torch.LongTensor(batch.action).unsqueeze(1)
        next_states = torch.FloatTensor(batch.next_state)
        rewards = torch.FloatTensor(batch.reward)
        dones = torch.FloatTensor(batch.done)

        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # 使用平滑L1损失（比MSE更稳定）
        loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
        self.loss_history.append(loss.item())  # 记录损失值

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# 延长训练回合，增加早停机制
def train_agent(episodes=500, target_score=490):
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # 记录更多训练指标
    scores = []
    steps_per_episode = []
    epsilons = []
    shaped_rewards = []  # 记录塑造后的奖励
    recent_scores = deque(maxlen=100)  # 记录最近100回合的得分，判断是否收敛
    convergence_episode = None
    converged = False

    for e in range(episodes):
        state, _ = env.reset()
        score = 0
        steps = 0
        total_shaped_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 奖励塑造：给小角度和小位移额外奖励，加速学习
            angle = abs(state[2])
            position = abs(state[0])
            shaped_reward = reward - 0.1 * angle - 0.05 * position
            agent.memory.push(state, action, next_state, shaped_reward, done)

            agent.learn()
            state = next_state
            score += reward
            total_shaped_reward += shaped_reward
            steps += 1

            if done:
                # 每5回合更新一次目标网络
                if e % agent.target_update_freq == 0:
                    agent.update_target_net()
                print(f"回合 {e + 1}/{episodes}, 得分: {score}, 步数: {steps}, ε: {agent.epsilon:.3f}")
                scores.append(score)
                steps_per_episode.append(steps)
                epsilons.append(agent.epsilon)
                shaped_rewards.append(total_shaped_reward)
                recent_scores.append(score)

                # 检测收敛
                if not converged and len(recent_scores) == 100:
                    if np.mean(recent_scores) >= target_score:
                        converged = True
                        convergence_episode = e + 1
                        print(f"在第 {convergence_episode} 回合达到收敛标准！")
                break

        # 早停：如果最近100回合平均得分达标，提前结束训练
        if converged:
            break

    env.close()
    return {
        'scores': scores,
        'steps': steps_per_episode,
        'epsilons': epsilons,
        'shaped_rewards': shaped_rewards,
        'loss_history': agent.loss_history,
        'convergence_episode': convergence_episode,
        'agent': agent,
        'total_episodes': len(scores)
    }


# 计算移动平均值
def moving_average(values, window_size=10):
    """计算移动平均值，用于平滑曲线展示收敛趋势"""
    if len(values) < window_size:
        return [np.mean(values[:i + 1]) for i in range(len(values))]
    return [np.mean(values[i:i + window_size]) for i in range(len(values) - window_size + 1)]


# 展示训练成果
def plot_training_metrics(results):
    """可视化训练过程中的各项指标"""
    scores = results['scores']
    steps = results['steps']
    epsilons = results['epsilons']
    shaped_rewards = results['shaped_rewards']
    loss_history = results['loss_history']
    convergence_episode = results['convergence_episode']
    total_episodes = results['total_episodes']

    window_size = 10
    avg_scores = moving_average(scores, window_size)
    # 计算损失的移动平均（每100步）
    loss_window = 100
    if len(loss_history) >= loss_window:
        avg_loss = moving_average(loss_history, loss_window)
    else:
        avg_loss = loss_history

    # 创建一个2x3的子图布局
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
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

    # 2. 每回合步数
    axs[0, 1].plot(steps, label='每回合步数')
    if convergence_episode:
        axs[0, 1].axvline(x=convergence_episode - 1, color='g', linestyle='--',
                          label=f'收敛点: 第{convergence_episode}回合')
    axs[0, 1].set_xlabel('回合数')
    axs[0, 1].set_ylabel('步数')
    axs[0, 1].set_title('每回合持续步数')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 3. 探索率变化
    axs[0, 2].plot(epsilons, label='ε值', color='orange')
    axs[0, 2].set_xlabel('回合数')
    axs[0, 2].set_ylabel('探索率 (ε)')
    axs[0, 2].set_title('探索率衰减曲线')
    axs[0, 2].legend()
    axs[0, 2].grid(True)
    axs[0, 2].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 4. 奖励塑造效果
    axs[1, 0].plot(shaped_rewards, label='塑造后的奖励', color='purple')
    axs[1, 0].set_xlabel('回合数')
    axs[1, 0].set_ylabel('总塑造奖励')
    axs[1, 0].set_title('每回合塑造奖励总和')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    axs[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 5. 损失函数变化
    axs[1, 1].plot(range(len(avg_loss)), avg_loss, label=f'每{loss_window}步平均损失', color='red')
    axs[1, 1].set_xlabel('训练步数 (x100)')
    axs[1, 1].set_ylabel('损失值')
    axs[1, 1].set_title('损失函数变化趋势')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # 6. 得分分布直方图
    axs[1, 2].hist(scores, bins=min(20, len(scores)), alpha=0.7, color='blue')
    axs[1, 2].axvline(x=np.mean(scores), color='r', linestyle='--', label=f'平均值: {np.mean(scores):.2f}')
    axs[1, 2].set_xlabel('得分')
    axs[1, 2].set_ylabel('频数')
    axs[1, 2].set_title('得分分布直方图')
    axs[1, 2].legend()
    axs[1, 2].grid(True, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # 打印关键统计信息
    print("\n===== 训练成果统计 =====")
    print(f"总训练回合数: {total_episodes}")
    print(f"最高得分: {max(scores)}")
    print(f"最低得分: {min(scores)}")
    print(f"最后10回合平均得分: {np.mean(scores[-10:]):.2f}")
    print(f"最后100回合平均得分: {np.mean(scores[-min(100, len(scores)):]):.2f}")
    print(f"总平均得分: {np.mean(scores):.2f}")
    print(f"得分标准差: {np.std(scores):.2f} (值越小表示性能越稳定)")
    if convergence_episode:
        print(f"达到收敛的回合数: {convergence_episode}")
    else:
        print("在训练过程中未达到收敛标准")


# 多次测试智能体并统计结果
def test_agent(agent, num_tests=5):
    """多次测试智能体并返回统计结果"""
    env = gym.make("CartPole-v1", render_mode="human")
    test_scores = []
    test_steps = []

    print(f"\n===== 开始智能体测试 ({num_tests}次) =====")
    for i in range(num_tests):
        state, _ = env.reset()
        score = 0
        steps = 0
        while True:
            action = agent.select_action(state)  # 使用训练好的策略
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            score += reward
            steps += 1
            if terminated or truncated:
                print(f"测试 {i + 1}/{num_tests} - 得分: {score}, 步数: {steps}")
                test_scores.append(score)
                test_steps.append(steps)
                break

    env.close()

    # 打印测试统计结果
    print("\n===== 测试结果统计 =====")
    print(f"测试次数: {num_tests}")
    print(f"平均得分: {np.mean(test_scores):.2f}")
    print(f"最高得分: {max(test_scores)}")
    print(f"最低得分: {min(test_scores)}")
    print(f"得分标准差: {np.std(test_scores):.2f}")

    # 绘制测试结果条形图
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, num_tests + 1), test_scores, color='skyblue')
    plt.axhline(y=np.mean(test_scores), color='r', linestyle='--', label=f'平均得分: {np.mean(test_scores):.2f}')
    plt.xlabel('测试次数')
    plt.ylabel('得分')
    plt.title(f'{num_tests}次测试得分分布')
    plt.xticks(range(1, num_tests + 1))
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.show()

    return {
        'test_scores': test_scores,
        'mean_score': np.mean(test_scores),
        'max_score': max(test_scores),
        'min_score': min(test_scores),
        'std_score': np.std(test_scores)
    }


if __name__ == "__main__":
    # 训练智能体
    training_results = train_agent(episodes=500, target_score=490)

    # 展示训练指标
    plot_training_metrics(training_results)

    # 测试训练好的智能体（多次测试取平均）
    test_results = test_agent(training_results['agent'], num_tests=5)
