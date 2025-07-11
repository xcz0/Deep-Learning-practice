# Deep-Learning-practice
使用pytorch与lightening框架实现经典深度学习算法

这个项目是一个 PyTorch Lightning 和 Hydra 的模板，旨在为机器学习实验提供一个用户友好的起点。 以下是该项目的使用方法解析：

### **快速上手**

1.  **克隆项目**：
    ```bash
    git clone https://github.com/ashleve/lightning-hydra-template
    cd lightning-hydra-template
    ```

2.  **创建 Conda 环境 (可选)**：
    ```bash
    conda create -n myenv python=3.9
    conda activate myenv
    ```

3.  **安装依赖**：
    首先根据你的环境安装 PyTorch，然后安装项目所需的其他库。
    ```bash
    pip install -r requirements.txt
    ```

4.  **运行训练**：
    该模板包含一个 MNIST 分类的示例。直接运行以下命令即可开始训练：
    ```bash
    python src/train.py
    ```

### **核心功能和使用方法**

#### **通过命令行覆盖配置**
你可以直接在命令行中覆盖任何配置参数，例如更改训练的 `epoch` 数或优化器的学习率：
```bash
python src/train.py trainer.max_epochs=20 model.optimizer.lr=1e-4
```

#### **在不同设备上训练**
- **CPU**: `python src/train.py trainer=cpu`
- **单 GPU**: `python src/train.py trainer=gpu`
- **多 GPU (DDP)**: `python src/train.py trainer=ddp trainer.devices=4`
- **TPU**: `python src/train.py +trainer.tpu_cores=8`

#### **混合精度训练**
使用 PyTorch 原生的自动混合精度 (AMP) 进行训练：
```bash
python src/train.py trainer=gpu +trainer.precision=16
```

#### **使用不同的日志记录器**
项目支持 Tensorboard, W&B, Neptune, Comet, MLFlow 等多种日志工具。你可以在 `configs/logger/` 目录下进行配置，并通过命令行指定使用哪个记录器：
```bash
python src/train.py logger=wandb
```

#### **运行实验配置**
你可以在 `configs/experiment/` 目录下创建不同的实验配置文件，并通过以下方式运行：
```bash
python src/train.py experiment=example
```

#### **调试**
项目提供了多种调试模式：
- **默认调试 (只运行 1 个 epoch)**: `python src/train.py debug=default`
- **快速开发运行 (只运行 1 个 batch)**: `python src/train.py debug=fdr`
- **分析器模式**: `python src/train.py debug=profiler`
- **过拟合单个 batch**: `python src/train.py debug=overfit`

#### **从断点续训**
从指定的检查点 (`.ckpt` 文件) 恢复训练：
```bash
python src/train.py ckpt_path="/path/to/ckpt/name.ckpt"
```

#### **评估模型**
在测试集上评估一个已经训练好的模型：
```bash
python eval.py ckpt_path="/path/to/ckpt/name.ckpt"
```

#### **超参数搜索**
- **网格搜索**:
  ```bash
  python train.py -m data.batch_size=32,64,128 model.lr=0.001,0.0005
  ```
- **使用 Optuna进行超参数搜索**:
  ```bash
  python train.py -m hparams_search=mnist_optuna experiment=example
  ```

### **项目结构**
- `configs/`: Hydra 的配置文件目录，包含了数据、模型、训练器等的所有配置。
- `src/`: 源代码目录，包含了模型 (`models`)、数据处理 (`data`) 和训练脚本 (`train.py`, `eval.py`)。
- `logs/`: 训练过程中生成的日志、模型断点等文件的存放目录。
- `notebooks/`: 用于数据探索等的 Jupyter Notebooks。

### **基本工作流程**

1.  在 `src/models/` 中编写你的 PyTorch Lightning 模块。
2.  在 `src/data/` 中编写你的 PyTorch Lightning 数据模块。
3.  在 `configs/experiment/` 中编写你的实验配置文件，指定模型和数据模块的路径及相关参数。
4.  运行训练脚本，并指定你的实验配置：`python src/train.py experiment=experiment_name.yaml`。