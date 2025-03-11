#!/usr/bin/env python3
"""
Enhanced BMX Analyzer: Comprehensive Soccer Betting Pattern Detection Tool

A sophisticated Python script designed to analyze soccer betting data 
and identify profitable betting patterns through advanced statistical techniques.

Key Features:
- AI Probability Band Analysis
- Flexible Pattern Detection with Gap Tolerance
- Comprehensive Statistical Validation
- Trend Analysis and Reporting
"""

import os
import sys
import json
import argparse
import logging
import warnings
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set, Union

# Scientific Computing and Data Analysis
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('BMXAnalyzer')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
np.random.seed(42)


#############################################
# BMXConfig Class
#############################################

class BMXConfig:
    """
    Configuration management for the BMX Analyzer.
    Handles system-wide settings, parameter validation, and model group configurations.
    """
    def __init__(self, data_path: str, output_path: str, config_file: Optional[str] = None):
        """
        Initialize configuration with data and output paths.
        
        Args:
            data_path (str): Path to directory containing input files.
            output_path (str): Path for output files and reports.
            config_file (str, optional): Path to model groups configuration file.
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.config_file = Path(config_file) if config_file else self.data_path / "model_groups.txt"
        
        # Analysis Configuration Parameters
        self.min_sample_size = 10
        self.min_sample_size_by_phase = 5
        self.min_pos_roi = 0.10  # 10% positive ROI threshold
        self.min_neg_roi = -0.20  # -20% negative ROI threshold
        self.season_decay_factor = 0.8
        self.min_pattern_width = 3
        self.max_pattern_gaps = 2
        self.pattern_gap_tolerance = 0.05
        self.phase_combination_min_improvement = 0.02
        self.bootstrap_iterations = 1000
        self.confidence_level = 0.95
        
        # Staking configurations - explicit naming
        self.staking_types = {
            "BACK": {
                "positive": ["MDS", "1US"],   # For positive ROI on BACK models
                "negative": ["GOM", "LAY1US"] # For negative ROI on BACK models
            },
            "LAY": {
                "positive": ["MDS", "1US"],   # For positive ROI on LAY models
                "negative": ["MDS", "1US"]    # For negative ROI on LAY models
            }
        }
        
        # Create output directories
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.temp_dir = self.output_path / "temp"
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Load model groups from the configuration file
        self.model_files = self._load_model_groups()
        
        logger.info(f"Initialized BMX Analyzer with data path: {self.data_path}")
        logger.info(f"Output directory: {self.output_path}")
    
    def _load_model_groups(self) -> Dict[str, List[str]]:
        """
        Load model groups from configuration file.
        
        Returns:
            Dict[str, List[str]]: Mapping of model names to file lists.
        """
        model_groups = {}
        if not self.config_file.exists():
            logger.warning(f"Configuration file not found: {self.config_file}")
            return model_groups
        
        try:
            with open(self.config_file, 'r') as f:
                for i, line in enumerate(f, 1):
                    files = [f.strip() for f in line.strip().split(',') if f.strip()]
                    if files:
                        model_name = f"Model_{i}"
                        model_groups[model_name] = files
            logger.info(f"Loaded {len(model_groups)} model groups from configuration")
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
        return model_groups
    
    def normalize_ai_percentages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize AI percentage columns to ensure consistent representation.
        Handles values as either decimals (0-1) or percentages (0-100).
        
        Args:
            df (pd.DataFrame): Input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with normalized AI percentages.
        """
        df = df.copy()
        ai_columns = [col for col in df.columns if 'ai' in str(col).lower()]
        logger.debug(f"Normalizing AI percentage columns: {ai_columns}")
        
        for col in ai_columns:
            # Check if column exists before processing
            if col in df.columns:
                # Proper normalization handling both decimal and percentage formats
                df[col] = df[col].apply(
                    lambda x: x * 100 if isinstance(x, (int, float)) and 0 <= x <= 1 else 
                             (x if isinstance(x, (int, float)) else None)
                )
                
                # Validate the range after normalization
                if df[col].max() > 100 or df[col].min() < 0:
                    logger.warning(f"Column {col} contains values outside the 0-100 range after normalization")
        
        return df
