# DFODE-Kit 智能 Agent 用户指南

本指南旨在规范 DFODE-kit 智能 Agent 的操作流程。该 Agent 旨在通过自然语言交互，自主完成燃烧模拟的工况设置、数据生成、模型训练及验证任务。

## 1. 前置准备 (Prerequisites)

在首次使用 Agent 之前，请确保已将 `dfode-kit` 安装至指定的 Conda 环境中，以便 Agent 能够调用最新的代码库。

### DFODE-kit 安装步骤
在终端中执行以下命令（仅需执行一次）：

```bash
conda run -n dfode_env pip install -e .
```

## 2. 启动与初始化 (Initialization)

为确保 Agent 能够正确索引文件路径并加载环境，每次会话请遵循以下初始化步骤。

1.  **工作目录**：请始终在项目根目录下启动 Agent 会话：
    ```bash
    cd /mnt/d/df-data/project/DFODE-kit/
    ```

2.  **加载上下文 (Context Loading)**：
    *   在与 Gemini CLI（或其他 Coding Agent）开启对话时，**必须首先提供操作手册**。
    *   **操作**：将项目根目录下的 `agent_skills.md` 文件内容复制并发送给 Agent，或作为附件上传。
    *   **提示词示例**：
        > “你好，你是 DFODE-kit 的操作专家。附件是你的操作手册 (`@agent_skills.md`)。请仔细阅读并严格遵守其中的环境配置规则、物理工况匹配逻辑以及安全协议。确认后，我将向你下达具体的燃烧建模任务。”

## 3. 任务下达规范 (Task Instruction)

用户无需指定具体的 CLI 命令或编写脚本。请直接描述**目标物理工况**和**应用需求**，Agent 将自动将其转化为技术参数。

*   **任务示例**：
    > “我现在需要训练一个用于二维预混HIT火焰模拟的模型，燃料为CH4，氧化剂为Air，当量比1.0，压力1atm，预混气温度300K，化学反应机理使用drm19机理。模拟步长1e-06，写出间隔1e-05，共写出10个结果。”
    > “我现在需要训练一个用于二维预混HIT火焰模拟的模型，燃料为H2，氧化剂为Air，当量比0.8，压力1atm，预混气温度300K，化学反应机理使用Burke机理。模拟步长1e-06，写出间隔1e-05，共写出10个结果。”

### 推荐指令结构
指令应包含以下核心要素：
*   **应用场景**：描述目标系统（如：航空发动机、燃气轮机、火箭发动机等）。
*   **工况参数**：
    *   **燃料 (Fuel)**：化学分子式（如 H2, CH4, n-C12H26）。
    *   **氧化剂 (Oxidizer)**：具体成分（如 Air, Pure O2）。
    *   **压力 (Pressure)**：具体数值或范围（如 1 atm, 30 bar）。
    *   **温度 (Temperature)**：入口温度或预热温度（如 300 K, 600 K）。
    *   **当量比 (Equivalence Ratio)**：关注的范围（如 0.5 至 1.2）。
*   **机理文件 (Mechanism)**：指定使用的化学反应机理文件路径。

## 4. 自动化执行流程 (Execution Process)

接收到指令后，Agent 将自主执行以下标准化全闭环流程：

1.  **物理映射与规划 (Physics Mapping)**：
    *   分析用户需求（如：高压、贫燃、预混/非预混）。
    *   基于物理特征选择最匹配的低维典型算例模板（Canonical Case Template）。
    *   进行参数换算（如将 atm 转换为 Pa）。

2.  **脚本生成与安全执行 (Scripting & Safety)**：
    *   调用 `DFODEAgentInterface` 编写自动化 Python 脚本。
    *   **安全协议**：
        *   自动创建带有**时间戳**的动态工作目录（如 `runs/20260121_140000_H2_Air_...`），严禁使用静态路径。
        *   严禁修改 `canonical_cases/` 下的模板文件。
    *   **环境管理**：使用 `conda run -n dfode_env` 封装执行命令，并自动加载 OpenFOAM 和 DeepFlame 的环境变量。
    *   **日志记录**：
        *   `execution.log`：记录任务宏观流程（Setup, Run, Train, Validate）。
        *   `train.log`：记录训练过程中的 Loss 曲线（CSV 格式）。

3.  **模拟与数据处理 (Simulation & Processing)**：
    *   **算例运行**：初始化算例文件（Setup）并调用 CFD 求解器（Run）。系统具备**鲁棒执行机制**：优先尝试并行计算，若检测到挂起或失败，自动切换为串行计算。
    *   **数据采样 (Sampling)**：对模拟结果进行热化学状态采样，生成 HDF5 原始数据集 (`data_raw.h5`)。
    *   **数据增强 (Augmentation) [必要步骤]**：Agent 将调用 `augment_data` 接口，通过随机微扰生成 `data_augmented.npy`。
    *   **数据标注 (Labeling) [必要步骤]**：调用 `label_data` 接口，计算增强数据的化学反应源项，生成 `data_labeled.npy`。
    *   **数据分割 (Splitting) [必要步骤]**：
        *   系统执行严格的 **8:1:1 划分**。
        *   **Training Set (80%)**：用于模型权重更新 (`*_train.npy`)。
        *   **Validation Set (10%)**：用于训练过程中的过拟合监控 (`*_val.npy`)。
        *   **Test Set (10%)**：**完全隔离**，仅用于最终的先验精度测试 (`*_test_unseen.npy`)。

4.  **模型训练 (Training)**：
    *   Agent 调用 `train_model` 接口，读取训练集进行梯度下降，读取验证集计算 `Val_Loss`。
    *   最终生成 `.pt` 模型文件。

5.  **模型验证 (Verification)**：
    *   **先验测试 (Priori Test)**：
        *   使用隔离的 **测试集** (`*_test_unseen.npy`) 进行离线推理。
        *   计算预测源项与真实值的误差 (RMSE)，评估模型的数学拟合精度。
    *   **后验测试 (Posteriori Test)**：
        *   自动搭建验证算例 (`posteriori_test/`)，将训练好的模型植入 OpenFOAM 求解器。
        *   运行实际 CFD 模拟，验证模型在流动耦合下的稳定性和物理准确性（如火焰传播速度）。

6.  **结果交付与报告 (Reporting)**：
    *   **自动化绘图**：
        *   `loss_curve.png`：展示 Training Loss 与 Validation Loss 的收敛曲线。
        *   `data_coverage.png`：展示训练数据在相空间（如 T vs Fuel）中对原始采样点的覆盖情况。
    *   **Markdown 报告**：
        *   在任务根目录下生成 `report.md`。
        *   汇总任务参数、训练 Epochs、最终 Loss、验证集 RMSE 以及上述可视化图表。

## 5. 输出文件结构 (Output Artifacts)

任务执行完成后，`runs/YYYYMMDD_.../` 目录下将包含：

*   **数据文件**：
    *   `data_raw.h5`: OpenFOAM 原始采样数据。
    *   `data_labeled_train.npy`: 训练集。
    *   `data_labeled_val.npy`: 验证集。
    *   `data_labeled_test_unseen.npy`: 测试集。
*   **模型文件**：
    *   `model.pt`: 训练好的 Neural ODE 模型（可直接部署）。
*   **日志与脚本**：
    *   `execution.log`: 全流程执行日志。
    *   `train.log`: 训练 Loss 数据 (CSV)。
    *   `run_task.py`: Agent 生成的可执行脚本。
*   **报告与验证**：
    *   `report.md`: 任务最终报告。
    *   `images/`: 包含 `loss_curve.png` 和 `data_coverage.png`。
    *   `posteriori_test/`: 后验验证算例目录（包含 OpenFOAM 运行结果）。
