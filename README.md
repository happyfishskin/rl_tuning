
# CUDA Auto-Tuning with Reinforcement Learning

本專案結合 CUDA 核心調優與強化學習（Reinforcement Learning, RL），使用 DQN 模型自動選擇最佳的 CUDA block/grid 配置，達成效能優化的目標。

## 專案架構

```
.
├── cuda_autotune_env.py       # 自定義 RL 環境，符合 OpenAI Gym 標準
├── train_rl_model.py          # 訓練 DQN 模型
├── test_rl.py                 # 使用 DQN 訓練 CartPole 環境的簡單範例
├── test.py                    # 載入訓練好的 DQN 模型並驗證效能
├── kernel_exec (需自行編譯)  # CUDA 執行檔，根據 block/grid 執行測試
```

## 功能說明

### 1`cuda_autotune_env.py`

- 定義 `CudaAutoTuneEnv` 環境。
- 每個動作代表一組 `(Block Size, Grid Size)`。
- 使用 `subprocess` 執行外部 `kernel_exec`，並回傳執行時間作為 reward（越快 reward 越高）。

### 2 `train_rl_model.py`

- 使用 Stable Baselines3 的 DQN 模型訓練 `CudaAutoTuneEnv`。
- 訓練完成後儲存模型為 `cuda_autotune_agent.zip`。

### 3 `test_rl.py`

- CartPole 環境的 DQN 簡單範例（非核心功能，可作為 DQN 入門參考）。

### 4 `test.py`

- 載入 `cuda_autotune_agent`，在 `CudaAutoTuneEnv` 中執行 10 次測試。
- 輸出每次選擇的 `(Block, Grid)` 與對應的執行時間。

## 環境需求

- Python 3.8+
- CUDA 12.2
- NVIDIA GPU（Compute Capability 6.1+）
- 必要套件：
  ```bash
  pip install stable-baselines3 gymnasium numpy
  ```

## 使用方式

### 1 準備 CUDA 執行檔

請自行撰寫並編譯一個 `kernel_exec` 可執行檔，其功能為根據輸入的 `(block_size, grid_size)` 執行 CUDA kernel 並輸出執行時間。範例如下：

```bash
./kernel_exec 128 64
```

> `kernel_exec` 需回傳正常的 exit code，且執行時間會由外部 `time` 測量。

### 2 訓練 DQN 模型

```bash
python train_rl_model.py
```

訓練過程會將模型儲存為 `cuda_autotune_agent.zip`，並於 `tensorboard_logs/` 中紀錄訓練過程。

### 3 測試模型效能

```bash
python test.py
```

範例輸出：

```
[Step 1] 選擇參數：Block Size=128, Grid Size=64，執行時間=0.01500 ms
[Step 2] 選擇參數：Block Size=64, Grid Size=128，執行時間=0.02000 ms
...
```

### 4 使用 CartPole 範例（非必要）

```bash
python test_rl.py
```

此程式會訓練並測試 DQN 模型於 `CartPole-v1` 環境。

## 計劃目標

- 結合 RL 與 CUDA Kernel 調參，實現自動化效能優化。
- 可拓展至其他 CUDA kernel 調優任務，例如矩陣運算、圖像處理等。
