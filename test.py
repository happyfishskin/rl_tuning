"""
功能：

載入訓練好的模型 cuda_autotune_agent。

在 CudaAutoTuneEnv 中執行 10 次測試步驟。

每次印出 RL 選擇的參數組合與對應的執行時間。

用途：

驗證 RL 模型是否能有效預測好參數，進而提升效能。
"""


from stable_baselines3 import DQN
from cuda_autotune_env import CudaAutoTuneEnv

def main():
    # 初始化環境與模型
    env = CudaAutoTuneEnv()
    model = DQN.load("cuda_autotune_agent")

    # 初始觀察值
    obs = env.reset()

    print("開始測試 RL 模型自動選擇 CUDA 參數...\n")

    for i in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        # 從 action 反推參數組合
        block_idx = action % len(env.block_size_options)
        grid_idx = action // len(env.block_size_options)
        block = env.block_size_options[block_idx]
        grid = env.grid_size_options[grid_idx]

        print(f"[Step {i+1}] 選擇參數：Block Size={block}, Grid Size={grid}，執行時間={-reward:.2f} ms")

        if done:
            obs = env.reset()

if __name__ == "__main__":
    main()
