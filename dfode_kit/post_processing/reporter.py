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
            plt.semilogy(df['Epoch'], df['TotalLoss'], label='Total Loss', linewidth=2)
            
            # Optional: Plot component losses if they exist and aren't negligible
            if 'Loss1' in df.columns:
                plt.semilogy(df['Epoch'], df['Loss1'], label='L1: Prediction', linestyle='--', alpha=0.7)
            if 'Loss2' in df.columns:
                plt.semilogy(df['Epoch'], df['Loss2'], label='L2: Mass Cons', linestyle='--', alpha=0.7)
            if 'Loss3' in df.columns:
                # Loss3 is often scaled, might be large or small
                plt.semilogy(df['Epoch'], df['Loss3'], label='L3: Enthalpy', linestyle='--', alpha=0.7)

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
        Generates report.md in the task directory.
        """
        report_path = self.work_dir / "report.md"
        
        md_content = f"""# DFODE-kit Task Report

**Task Path:** `{self.work_dir.name}`
**Date:** {info.get('date', 'N/A')}
**Status:** {info.get('status', 'Completed')}

## 1. Task Summary
- **Fuel:** {info.get('fuel', 'N/A')}
- **Oxidizer:** {info.get('oxidizer', 'N/A')}
- **Equivalence Ratio (Phi):** {info.get('phi', 'N/A')}
- **Pressure:** {info.get('p', 'N/A')}
- **Temperature:** {info.get('t', 'N/A')}
- **Mechanism:** `{Path(info.get('mechanism', '')).name}`

## 2. Model Performance
- **Priori Test RMSE:** {info.get('rmse', 'N/A')}
- **Training Epochs:** {info.get('epochs', 'N/A')}
- **Final Loss:** {info.get('final_loss', 'N/A')}

"""

        if 'loss_curve' in images and images['loss_curve']:
            md_content += f"""## 3. Training Convergence
*(Loss over Epochs)*
![Loss Curve](images/{images['loss_curve']})

"""

        if 'data_coverage' in images and images['data_coverage']:
            md_content += f"""## 4. Dataset Analysis
*(Augmented Training Data vs Original Samples)*
![Data Coverage](images/{images['data_coverage']})
> **Blue:** Augmented Training Set (Bottom Layer)  
> **Red:** Original OpenFOAM Samples (Top Layer)

"""

        md_content += """## 5. Key Artifacts
| File | Description |
|------|-------------|
| `model.pt` | Trained Neural ODE Model |
| `data_raw.h5` | Raw Simulation Data |
| `data_labeled_train.npy` | Training Dataset |
| `train.log` | Training Metrics (CSV) |
| `execution.log` | Workflow Execution Log |
"""

        with open(report_path, 'w') as f:
            f.write(md_content)
            
        return str(report_path)
