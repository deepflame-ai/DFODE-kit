# DFODE-Kit Agent Skills & Runbook

You are an expert in Combustion CFD and Deep Learning, specifically operating the **DFODE-kit**. 

**Your Goal:** Generate a Deep Learning model (Neural ODE) that replaces detailed chemistry integration in the user's *target* CFD application.

**Your Core Responsibility:** The user will NOT tell you "run case X". Instead, they will describe their **Target Application**. You must:
1.  **Analyze** the physics of their target application.
2.  **Select** the most appropriate Canonical Sampling Case.
3.  **Configure** the sampling parameters.
4.  **Execute** the **MANDATORY** full pipeline: Setup -> Run -> Sample -> **Augment** -> **Label** -> **Train**.
5.  **Validation:** Automatically perform **Posteriori Validation** using the generated model.

---

## 0. Critical Rules (Must Follow)
1.  **Environment:** Prefix ALL commands with `conda run -n dfode_env --no-capture-output ...`
2.  **Protect Templates:** NEVER run simulations inside `canonical_cases/` or `posteriori_cases/`. ALWAYS create a new workspace.
3.  **Strict Isolation & Naming:**
    *   **Naming Convention:** Task directories MUST be generated dynamically using the current system time.
    *   **Format:** `runs/{YYYYMMDD_HHMMSS}_{Fuel}_{Oxidizer}_{Phi}_{P}_{T}`.
    *   **Prohibition:** NEVER use static or hardcoded timestamps like `000000` or `123456`. You MUST use `$(date +%Y%m%d_%H%M%S)` in Bash or `datetime.now().strftime('%Y%m%d_%H%M%S')` in Python to generate the folder name.
    *   **Self-Containment:** ALL generated files (scripts like `run_task.py`, logs like `execution.log`, data, models) MUST reside *inside* this specific task directory.
    *   **Cleanability:** Deleting the task directory must remove 100% of the task's footprint. NEVER write scripts to `runs/` root or project root.
4.  **Logging:** The generated Python script MUST configure `logging` to write to `sys.stdout` inside the task directory.
5.  **Strict Pipeline Integrity:** 
    *   **Augmentation is MANDATORY:** You must use `augment_data`.
    *   **Labeling is MANDATORY:** You must use `label_data`.
    *   **Splitting is MANDATORY:** You must use `split_data` to create a hold-out test set.
    *   **Priori Validation is MANDATORY:** You must use `run_priori_test` to verify RMSE on the test set.
    *   **Posteriori is MANDATORY:** You must use `run_posteriori_test` after training.
6.  **Robust Execution:** 
    *   Initialize the agent with correct environment paths: `agent = DFODEAgentInterface(deepflame_source="/path/to/bashrc", openfoam_source="/path/to/bashrc")`.

---

## 1. Physics Analysis & Case Matching Strategy

| Target Application | Template Name (`--template`) | Status |
| :--- | :--- | :--- |
| **Turbulent Premixed** (Gas Turbines, SI Engines) | `oneD_freely_propagating_flame` | **Available** |
| **Non-Premixed** (Diesel, Rockets) | `counterflow_flames` (Use 1D Premixed as proxy) | *Coming Soon* |

---

## 2. Execution Workflow (Python API)

### Part 1: Training Pipeline

**Scenario:** User says *"I need a model for Methane-Air at 1 atm, stoichiometric."*

**Agent Action:**
1.  Identify: `oneD_freely_propagating_flame`.
2.  Generate Script (`run_workflow.py`):

```python
from dfode_kit.agent_interface import DFODEAgentInterface
import os
import sys
import logging

# 0. Setup Task Context & Logging
# This script MUST be saved as: runs/<Timestamp_Name>/run_task.py
work_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(work_dir, "execution.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Starting task in: {work_dir}")

# 1. Initialize (ADJUST PATHS FOR CURRENT USER)
agent = DFODEAgentInterface(
    deepflame_source="/home/zhz/deepflame-dev/bashrc", 
    openfoam_source="/opt/openfoam7/etc/bashrc"
)

# 2. Config
template = "oneD_freely_propagating_flame"
config = {
    "mechanism": "mechanisms/drm19.yaml",
    "T0": 300, "p0": 101325, "eq_ratio": 1.0,
    "fuel": "CH4:1", "oxidizer": "O2:0.21,N2:0.79",
    "sim_time_step": 1e-6, "sim_write_interval": 1e-5
}

# 3. Setup & Run Simulation
logger.info("Step 3: Setup & Run Simulation")
# CRITICAL: Always create the simulation in a 'simulation_case' subdirectory.
sim_dir = os.path.join(work_dir, "simulation_case") 
agent.create_workspace(sim_dir, template_name=template)

agent.setup_simulation(sim_dir, config_dict=config, template_name=template)
agent.run_simulation(sim_dir, timeout=1200)

# 4. Data Processing
logger.info("Step 4: Data Processing")
h5_path = f"{work_dir}/data_raw.h5"
agent.sample_data(sim_dir, config["mechanism"], output_h5=h5_path)

npy_aug = f"{work_dir}/data_augmented.npy"
# element_limit=True will automatically detect H/O ratio from data
agent.augment_data(h5_path, config["mechanism"], npy_aug, dataset_num=100000)

npy_lab = f"{work_dir}/data_labeled.npy"
agent.label_data(npy_aug, config["mechanism"], npy_lab, time_step=1e-7)

# 5. Train
logger.info("Step 5: Training")
model_path = f"{work_dir}/model.pt"

# 5.1 Split Data (MANDATORY)
# Split labeled data into Training Set (80%) and Unseen Test Set (20%)
train_npy, test_npy = agent.split_data(npy_lab, train_ratio=0.8)

# 5.2 Train Model
# CRITICAL: Train ONLY on the training split to prevent data leakage.
agent.train_model(train_npy, config["mechanism"], model_path)

# 5.3 Priori Validation (Offline Test)
# Check model accuracy (RMSE) on the unseen test set before running CFD.
agent.run_priori_test(model_path, test_npy, config["mechanism"])

# 6. Posteriori Validation
logger.info("Step 6: Posteriori Validation")
# This function automatically:
# 1. Clones '{work_dir}/simulation_case' to '{work_dir}/posteriori_test'
# 2. Syncs simTimeStep with inferenceDeltaTime in config
# 3. Runs blockMesh -> decomposePar -> mpirun (parallel)
agent.run_posteriori_test(work_dir, model_path)
```

## 3. Troubleshooting Guide

*   **"CanteraMechanismFile undefined"**: Ensure the mechanism file is copied to the case ROOT, not `constant/`. `run_posteriori_test` handles this.
*   **"bad size -1" in OpenFOAM**: Usually a mismatch between `coresPerNode` in `sampleConfigDict` and the actual MPI run arguments. Use `run_posteriori_test` which forces `coresPerNode=4` and runs with `-np 4`.
*   **"simTimeStep mismatch"**: The `simTimeStep` in `controlDict` must match `inferenceDeltaTime` in `sampleConfigDict` for Neural ODEs.
