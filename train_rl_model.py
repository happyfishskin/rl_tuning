from stable_baselines3 import DQN
from cuda_autotune_env import CudaAutoTuneEnv

def main():
    # 初始化自定義環境
    env = CudaAutoTuneEnv()

    # 建立 DQN 模型
    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        tensorboard_log="./tensorboard_logs/"
    )

    # 開始訓練
    model.learn(total_timesteps=50000, log_interval=1, progress_bar=True)

    # 儲存模型
    model.save("cuda_autotune_agent")
    print("訓練完成，模型已儲存為 cuda_autotune_agent")

if __name__ == "__main__":
    main()
