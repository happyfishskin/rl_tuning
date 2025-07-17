import gymnasium as gym
from stable_baselines3 import DQN

def main():
    # 使用 Gymnasium 環境
    env = gym.make("CartPole-v1")

    # 建立 DQN 模型
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)
    model.save("dqn_cartpole")

    print("CartPole 訓練完成，開始測試...\n")

    obs, _ = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            print(f"Episode 結束，重設環境 (Step {i})")
            obs, _ = env.reset()

    print("測試結束。")

if __name__ == "__main__":
    main()
