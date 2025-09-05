import gymnasium as gym
import pygame  # 用于键盘输入

# 创建环境，开启可视化
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

# 初始化 pygame（处理键盘输入）
pygame.init()
clock = pygame.time.Clock()

# 初始化动作变量，避免未定义错误
action = 0  # 初始动作设为左移（0）

while True:
    # 处理退出事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            pygame.quit()
            exit()

    # 核心修改：检测持续按键状态（替换原来的KEYDOWN事件检测）
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action = 0  # 左移
    elif keys[pygame.K_RIGHT]:
        action = 1  # 右移
    # 注意：CartPole环境没有"不动"的动作，松开按键时会保持最后一次的动作方向

    # 执行动作，获取反馈
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"状态: {observation}, 奖励: {reward}")

    # 若游戏结束（杆子倒下或出界），重置环境
    if terminated or truncated:
        print("游戏结束！重置环境...")
        observation, info = env.reset()

    clock.tick(10)  # 控制帧率，数值越大速度越快

# 清理资源（理论上不会执行到这里，因为循环是无限的）
env.close()
pygame.quit()
