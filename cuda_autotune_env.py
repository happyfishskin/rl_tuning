"""
功能：

實作符合 gym.Env 標準的強化學習環境 CudaAutoTuneEnv。

每個 action 對應一組 (block_size, grid_size)。

使用 subprocess 呼叫 kernel_exec，並擷取執行時間。

將 reward 設為負的執行時間（即：執行越快，reward 越高）。

observation 為 one-hot 編碼表示目前選擇的參數組合。

用途：

提供給 RL agent 用來互動、學習如何選擇最佳參數組合。
"""
import gym
import numpy as np
import subprocess
import time
from gym import spaces

class CudaAutoTuneEnv(gym.Env):
    def __init__(self):
        super(CudaAutoTuneEnv, self).__init__()

        # 可調整的 Block / Grid 值（根據你 CUDA kernel 實作設定）
        self.block_size_options = [32, 64, 128, 256]
        self.grid_size_options = [16, 32, 64, 128]

        # 動作空間：block index x grid index 組合
        self.action_space = spaces.Discrete(len(self.block_size_options) * len(self.grid_size_options))

        # 狀態空間（簡化為 one-hot of action index）
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int32)

        self.current_action = None
        self.exec_time = None

    def _decode_action(self, action_idx):
        block_idx = action_idx % len(self.block_size_options)
        grid_idx = action_idx // len(self.block_size_options)
        return self.block_size_options[block_idx], self.grid_size_options[grid_idx]

    def step(self, action):
        self.current_action = action
        block_size, grid_size = self._decode_action(action)

        # 呼叫 CUDA 執行檔，傳入 block/grid 作為參數
        try:
            start = time.time()
            result = subprocess.run(
                ["./kernel_exec", str(block_size), str(grid_size)],
                capture_output=True, text=True, timeout=10
            )
            end = time.time()

            # 嘗試擷取 kernel 執行時間（若程式有輸出）
            exec_time_ms = end - start
            if result.returncode != 0:
                print(f"執行錯誤: {result.stderr}")
                exec_time_ms = 9999.0  # 失敗懲罰
        except Exception as e:
            print(f"例外錯誤: {e}")
            exec_time_ms = 9999.0

        # reward 為負的執行時間
        reward = -exec_time_ms
        obs = self._get_obs(action)
        done = True  # 單步完成（可改為 multi-step）

        return obs, reward, done, {}

    def _get_obs(self, action):
        obs = np.zeros(self.action_space.n, dtype=np.int32)
        obs[action] = 1
        return obs

    def reset(self):
        self.current_action = None
        self.exec_time = None
        return np.zeros(self.action_space.n, dtype=np.int32)

    def render(self, mode="human"):
        if self.current_action is not None:
            block_size, grid_size = self._decode_action(self.current_action)
            print(f"Action: Block={block_size}, Grid={grid_size}, Exec Time={self.exec_time} ms")

