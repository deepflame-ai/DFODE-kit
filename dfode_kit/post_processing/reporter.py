import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cantera as ct
from pathlib import Path
from typing import Dict, Optional, Any

# Configure logger for this module
logger = logging.getLogger(__name__)

class TaskReporter:
    def __init__(self, work_dir: str):
        self.work_dir = Path(work_dir)
        self.img_dir = self.work_dir / "images"
        self.img_dir.mkdir(exist_ok=True)
        
        # Set matplotlib style for publication quality
        plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
        # Ensure non-interactive backend
        plt.switch_backend('Agg')

    def _parse_fuel_name(self, fuel_str: str) -> str:
        """
        Extracts the primary fuel species name from a string like 'H2:1' or 'CH4:1.0'.
        Assumes the part before the first ':' is the species name.
        """
        if ":" in fuel_str:
            return fuel_str.split(":")[0].strip()
        return fuel_str.strip()

    def plot_data_coverage(self, raw_h5_path: str, train_npy_path: str, mech_path: str, fuel_config_str: str):
        """
        Plots the coverage of the augmented training set over the original sampled data.
        X-axis: Temperature (Index 0)
        Y-axis: Fuel Mass Fraction (Index 2 + Species Index)
        
        Layers: 
        - Bottom: Augmented data (Blue, scatter)
        - Top: Raw data (Red, scatter)
        """
        logger.info("Generating data coverage plot...")
        output_path = self.img_dir / "data_coverage.png"
        
        try:
            # 1. Determine Fuel Column Index
            # Ensure mech_path is absolute or correct
            if not Path(mech_path).exists():
                # Try finding it in mechanisms folder if DFODE_ROOT is available
                # But here we assume the caller provided a valid path or we fail gracefully
                pass 

            fuel_name = self._parse_fuel_name(fuel_config_str)
            
            try:
                gas = ct.Solution(str(mech_path))
                fuel_idx = gas.species_index(fuel_name)
                if fuel_idx == -1:
                    raise ValueError(f"Fuel species '{fuel_name}' not found in mechanism.")
                col_idx = fuel_idx + 2 # T, P are 0, 1
                logger.info(f"Fuel '{fuel_name}' found at species index {fuel_idx}, data column {col_idx}")
            except Exception as e:
                logger.error(f"Failed to load mechanism or find fuel index: {e}")
                return None

            # 2. Load Data
            # Import h5_kit here to avoid circular deps if any, or just use what we need
            from dfode_kit.data_operations.h5_kit import get_TPY_from_h5
            
            raw_data = get_TPY_from_h5(raw_h5_path)
            train_data = np.load(train_npy_path)
            
            # 3. Use Full Datasets (No Downsampling)
            train_sample = train_data # Use full dataset as requested

            # 4. Plot
            plt.figure(figsize=(10, 6))
            
            # Bottom layer: Augmented/Train
            plt.scatter(train_sample[:, 0], train_sample[:, col_idx], 
                        c='blue', alpha=0.3, s=10, label='Augmented (Train)')
            
            # Top layer: Raw
            plt.scatter(raw_data[:, 0], raw_data[:, col_idx], 
                        c='red', alpha=0.6, s=10, label='Raw Sampled')
            
            plt.xlabel("Temperature (K)")
            plt.ylabel(f"Mass Fraction: {fuel_name}")
            plt.title(f"Data Coverage: Temperature vs {fuel_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Data coverage plot saved to {output_path}")
            return output_path.name
            
        except Exception as e:
            logger.error(f"Error plotting data coverage: {e}")
            return None

    def plot_loss_curve(self, log_path: str):
        """
        Plots the training loss curve from train.log (CSV format).
        """
        logger.info("Generating loss curve plot...")
        output_path = self.img_dir / "loss_curve.png"
        
        if not os.path.exists(log_path):
            logger.warning(f"Training log not found at {log_path}. Skipping loss plot.")
            return None
            
        try:
            # Read CSV
            df = pd.read_csv(log_path)
            
            if 'Epoch' not in df.columns or 'TotalLoss' not in df.columns:
                logger.error("train.log format invalid. Expected 'Epoch' and 'TotalLoss' columns.")
                return None
                
            plt.figure(figsize=(10, 6))
            
            # Plot Total Loss (Log Scale usually better for loss)
            plt.semilogy(df['Epoch'], df['TotalLoss'], label='Train Loss', linewidth=2)
            
            # Plot Validation Loss if available
            if 'ValLoss' in df.columns:
                # Use a rolling mean for validation loss to smooth it out if it's noisy, or just plot raw
                plt.semilogy(df['Epoch'], df['ValLoss'], label='Validation Loss', linewidth=2, linestyle='-')

            # Optional: Plot component losses (REMOVED as per user request)
            # if 'Loss1' in df.columns:
            #     plt.semilogy(df['Epoch'], df['Loss1'], label='L1: Prediction', linestyle='--', alpha=0.7)
            # if 'Loss2' in df.columns:
            #     plt.semilogy(df['Epoch'], df['Loss2'], label='L2: Mass Cons', linestyle='--', alpha=0.7)
            # if 'Loss3' in df.columns:
            #     plt.semilogy(df['Epoch'], df['Loss3'], label='L3: Enthalpy', linestyle='--', alpha=0.7)

            plt.xlabel("Epoch")
            plt.ylabel("Loss (Log Scale)")
            plt.title("Neural ODE Training Convergence")
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.2)
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Loss curve saved to {output_path}")
            return output_path.name
            
        except Exception as e:
            logger.error(f"Error plotting loss curve: {e}")
            return None

    def create_markdown_report(self, info: Dict[str, Any], images: Dict[str, str]) -> str:
        """
        Generates report.md in the task directory based on user requirements.
        
        Sections:
        1. Base Configuration (Task Settings)
        2. Data Processing & Analysis (Sizes, Split, Coverage Plot)
        3. Model Training & Validation (Loss Plot, Hyperparams, Metrics)
        4. Key Artifacts (File List)
        """
        report_path = self.work_dir / "report.md"
        
        # --- Section 1: Base Configuration ---
        md_content = f"""# DFODE-kit Task Report

**Task Path:** `{self.work_dir.name}`
**Date:** {info.get('date', 'N/A')}
**Status:** {info.get('status', 'Completed')}

## 1. Base Configuration
| Parameter | Value |
| :--- | :--- |
| **Fuel** | `{info.get('fuel', 'N/A')}` |
| **Oxidizer** | `{info.get('oxidizer', 'N/A')}` |
| **Equivalence Ratio ($\phi$)** | {info.get('phi', 'N/A')} |
| **Pressure** | {info.get('p', 'N/A')} Pa |
| **Temperature** | {info.get('t', 'N/A')} K |
| **Mechanism** | `{Path(info.get('mechanism', '')).name}` |

"""

        # --- Section 2: Data Processing & Analysis ---
        md_content += "## 2. Data Processing & Analysis\n\n"
        
        # Data Sizes Table
        md_content += "### 2.1 Dataset Statistics\n"
        md_content += "| Dataset Stage | Sample Count |\n"
        md_content += "| :--- | :--- |\n"
        md_content += f"| **Raw Sampling** | {info.get('size_raw', 'N/A')} |\n"
        md_content += f"| **Augmented** | {info.get('size_augmented', 'N/A')} |\n\n"
        
        # Split Info
        md_content += "### 2.2 Data Splitting\n"
        md_content += f"**Strategy:** {info.get('split_strategy', 'N/A')}\n\n"
        md_content += "| Split | Count |\n"
        md_content += "| :--- | :--- |\n"
        md_content += f"| **Training Set** (80%) | {info.get('size_train', 'N/A')} |\n"
        md_content += f"| **Validation Set** (10%) | {info.get('size_val', 'N/A')} |\n"
        md_content += f"| **Test Set** (10%) | {info.get('size_test', 'N/A')} |\n\n"

        # Coverage Plot
        if 'data_coverage' in images and images['data_coverage']:
            md_content += f"### 2.3 Data Coverage Visualization\n"
            md_content += "*(Augmented Training Data vs Original Samples)*\n"
            md_content += f"![Data Coverage](images/{images['data_coverage']})\n"
            md_content += "> **Blue:** Augmented Training Set (Bottom Layer)  \n"
            md_content += "> **Red:** Original OpenFOAM Samples (Top Layer)\n\n"

        # --- Section 3: Model Training & Validation ---
        md_content += "## 3. Model Training & Validation\n\n"
        
        # Loss Curve
        if 'loss_curve' in images and images['loss_curve']:
            md_content += "### 3.1 Training Convergence\n"
            md_content += "*(Loss over Epochs)*\n"
            md_content += f"![Loss Curve](images/{images['loss_curve']})\n\n"
            
        # Training Metadata
        md_content += "### 3.2 Training Configuration & Metrics\n"
        md_content += "| Metric | Value |\n"
        md_content += "| :--- | :--- |\n"
        md_content += f"| **Total Epochs** | {info.get('epochs', 'N/A')} |\n"
        md_content += f"| **Batch Size** | {info.get('batch_size', '20000 (Default)')} |\n"
        md_content += f"| **Learning Rate** | {info.get('learning_rate', '0.001 (Default)')} |\n"
        md_content += f"| **Final Training Loss** | {info.get('final_loss', 'N/A')} |\n"
        md_content += f"| **Priori Test RMSE** | **{info.get('rmse', 'N/A')}** |\n\n"

        # --- Section 4: Key Artifacts ---
        md_content += """## 4. Key Artifacts
*(Generated files in the task directory)*

| File | Description |
| :--- | :--- |
| `model.pt` | **Trained Neural ODE Model** (Ready for deployment) |
| `report.md` | This Report |
| `execution.log` | Task Workflow Log |
| `train.log` | Training Metrics (CSV) |
| `data_raw.h5` | Raw Sampling Data (HDF5) |
| `data_labeled_train.npy` | Training Dataset (Numpy) |
| `data_labeled_val.npy` | Validation Dataset (Numpy) |
| `data_labeled_test_unseen.npy` | Hold-out Test Dataset (Numpy) |
| `posteriori_test/` | Verification Case Directory |
"""

        with open(report_path, 'w') as f:
            f.write(md_content)
            
        return str(report_path)
