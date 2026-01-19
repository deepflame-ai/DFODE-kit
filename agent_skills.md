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
3.  **NO Root Pollution:** NEVER create logs, scripts (`.py`), or output files in the project root.
    *   ALWAYS use `agent.init_task(task_name)` to create a timestamped, isolated workspace.
4.  **Strict Pipeline Integrity:** 
    *   **Augmentation is MANDATORY:** You must use `augment_data`.
    *   **Labeling is MANDATORY:** You must use `label_data`.
    *   **Posteriori is MANDATORY:** You must use `run_posteriori_test` after training.
5.  **Robust Execution:** 
    *   Initialize the agent with correct environment paths if the user is not 'skylark': `agent = DFODEAgentInterface(deepflame_source="/path/to/bashrc", openfoam_source="/path/to/bashrc")`.

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

# 1. Initialize (ADJUST PATHS FOR CURRENT USER)
agent = DFODEAgentInterface(
    deepflame_source="/home/zhz/deepflame-dev/bashrc", 
    openfoam_source="/opt/openfoam7/etc/bashrc"
)

# 2. Config
work_dir = os.path.abspath("runs/TaskName")
template = "oneD_freely_propagating_flame"
config = {
    "mechanism": "mechanisms/drm19.yaml",
    "T0": 300, "p0": 101325, "eq_ratio": 1.0,
    "fuel": "CH4:1", "oxidizer": "O2:0.21,N2:0.79",
    "sim_time_step": 1e-6, "sim_write_interval": 1e-5
}

# 3. Setup & Run Simulation
# NOTE: The simulation case is created inside a subdirectory '{work_dir}/simulation_case'
agent.create_workspace(work_dir, template_name=template)
sim_dir = os.path.join(work_dir, "simulation_case")
agent.setup_simulation(sim_dir, config_dict=config, template_name=template)
agent.run_simulation(sim_dir, timeout=1200)

# 4. Data Processing
h5_path = f"{work_dir}/data_raw.h5"
agent.sample_data(sim_dir, config["mechanism"], output_h5=h5_path)

npy_aug = f"{work_dir}/data_augmented.npy"
# element_limit=True will automatically detect H/O ratio from data
agent.augment_data(h5_path, config["mechanism"], npy_aug, dataset_num=100000)

npy_lab = f"{work_dir}/data_labeled.npy"
agent.label_data(npy_aug, config["mechanism"], npy_lab, time_step=1e-7)

# 5. Train
model_path = f"{work_dir}/model.pt"
agent.train_model(npy_lab, config["mechanism"], model_path)
```

### Part 2: Posteriori Validation (CFD Test)

**Goal:** Verify the trained model in an actual OpenFOAM simulation.

**Strategy:** Do NOT manually configure the posteriori case. Reuse the physics from the sampling simulation.

**Agent Action:**
Append this to the script:

```python
# 6. Posteriori Validation
print("--- Step 7: Posteriori Validation ---")
# This function automatically:
# 1. Clones '{work_dir}/simulation_case' to '{work_dir}/posteriori_test'
# 2. Syncs simTimeStep with inferenceDeltaTime in config
# 3. Copies inference.py and model.pt
# 4. Sets Torch=on, GPU=on, Cores=4
# 5. Runs blockMesh -> decomposePar -> mpirun (parallel)
agent.run_posteriori_test(work_dir, model_path)
```

## 3. Troubleshooting Guide

*   **"CanteraMechanismFile undefined"**: Ensure the mechanism file is copied to the case ROOT, not just `constant/`. `run_posteriori_test` handles this.
*   **"bad size -1" in OpenFOAM**: Usually a mismatch between `coresPerNode` in `sampleConfigDict` and the actual MPI run arguments. Use `run_posteriori_test` which forces `coresPerNode=4` and runs with `-np 4`.
*   **"simTimeStep mismatch"**: The `simTimeStep` in `controlDict` must match `inferenceDeltaTime` in `sampleConfigDict` for Neural ODEs.
