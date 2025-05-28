import os
import shutil
import subprocess
from pathlib import Path

import pytest


class TestBasicCLI:
    """Test basic CLI functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self, synthetic_data, temp_output_dir):
        """Setup test environment."""
        self.data = synthetic_data
        self.output_dir = temp_output_dir
        self.save_dir = self.output_dir / "save_dir"
        self.save_dir.mkdir()
        
        # Store original working directory
        self.original_dir = os.getcwd()
        
        yield
        
        # Cleanup
        os.chdir(self.original_dir)
        for dir_to_clean in ["temp", "logs"]:
            if os.path.exists(dir_to_clean):
                shutil.rmtree(dir_to_clean)
    
    def test_simple_cli(self):
        """Test basic CLI command without folds."""
        cmd = f"normative -c {self.data['cov_path']} -a blr -r {self.data['resp_path']} save_dir={self.save_dir}"
        
        # Run command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        
        # Verify output files
        assert (self.save_dir / "model" / "normative_model.json").exists()
        assert (self.save_dir / "plots" / "centiles_response_var_0_fit_data_harmonized.png").exists()
        assert (self.save_dir / "results" / "centiles_fit_data.csv").exists()
    
    def test_cli_with_folds(self):
        """Test CLI command with cross-validation folds."""
        cmd = f"normative -c {self.data['cov_path']} -a blr -r {self.data['resp_path']} -k 3 save_dir={self.save_dir}"
        
        # Run command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        
        # Verify output files for each fold
        for fold in range(3):
            fold_dir = self.save_dir / "folds" / f"fold_{fold}"
            assert (fold_dir / "model" / "normative_model.json").exists()
            assert (fold_dir / "plots" / f"centiles_response_var_0_fit_data_fold_{fold}_train_harmonized.png").exists()
            assert (fold_dir / "results" / f"centiles_fit_data_fold_{fold}_train.csv").exists()
    
    def test_cli_with_batch_effects(self):
        """Test CLI command with batch effect correction."""
        cmd = f"normative -c {self.data['cov_path']} -a blr -r {self.data['resp_path']} -b {self.data['batch_path']} save_dir={self.save_dir}"
        
        # Run command
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        
        # Verify output files
        assert (self.save_dir / "model" / "normative_model.json").exists()
        assert (self.save_dir / "plots" / "centiles_response_var_0_fit_data_harmonized.png").exists()
        assert (self.save_dir / "results" / "centiles_fit_data.csv").exists() 