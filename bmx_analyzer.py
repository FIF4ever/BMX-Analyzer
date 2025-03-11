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
        
        return dfdef identify_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        Dynamically identify key columns in the DataFrame with comprehensive matching.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
        
        Returns:
            Dict[str, Optional[str]]: Mapping of semantic column names to actual column names.
        """
        # Initialize with all possible column types we need to identify
        column_mapping = {
            'league': None,
            'selection': None,
            'ai_pct': None,
            'phase': None,
            'season_number': None,
            'model_type': None,
            
            # Staking types - both ROI, stake and payout columns
            'mds_roi': None,
            'mds_stake': None,
            'mds_payout': None,
            
            '1us_roi': None,
            '1us_stake': None,
            '1us_payout': None,
            
            'gom_roi': None,
            'gom_stake': None, 
            'gom_payout': None,
            
            'lay1us_roi': None,
            'lay1us_stake': None,
            'lay1us_payout': None,
            
            # Additional columns
            'odds': None,
            'result': None,
            'date': None
        }
        
        # Print all columns for debugging
        logger.debug(f"All columns in dataframe: {df.columns.tolist()}")
        
        # First, try to print the first few rows to understand the data structure
        try:
            sample_data = df.head(3).to_dict()
            logger.debug(f"Sample data: {sample_data}")
        except Exception as e:
            logger.debug(f"Could not print sample data: {e}")
        
        # Extended map of keywords to column types with more variations
        column_patterns = {
            'league': ['league', 'comp', 'competition', 'division', 'div', 'lg'],
            'selection': ['selection', 'outcome', 'pick', 'bet', 'sel', 'choice', 'type'],
            'ai_pct': ['ai', '%', 'pct', 'percent', 'percentage', 'probability', 'prob', 'confidence'],
            'phase': ['phase', 'gameweek', 'week', 'stage', 'period', 'part'],
            'season_number': ['season', 'ssn', 'year', 'yr', 'sn'],
            'model_type': ['model', 'type', 'br', 'back/lay', 'back', 'lay', 'backlay'],
            
            # MDS columns
            'mds_roi': ['mds', 'roi', 'return', 'profit'],
            'mds_stake': ['mds', 'stake', 'unit', 'bet', 'wager'],
            'mds_payout': ['mds', 'payout', 'return', 'total', 'payment'],
            
            # 1US columns
            '1us_roi': ['1us', 'roi', 'return', 'profit'],
            '1us_stake': ['1us', 'stake', 'unit', 'bet', 'wager'],
            '1us_payout': ['1us', 'payout', 'return', 'total', 'payment'],
            
            # GOM columns
            'gom_roi': ['gom', 'roi', 'return', 'profit'],
            'gom_stake': ['gom', 'stake', 'unit', 'bet', 'wager'],
            'gom_payout': ['gom', 'payout', 'return', 'total', 'payment'],
            
            # LAY1US columns
            'lay1us_roi': ['lay1us', 'lay', '1us', 'roi', 'return', 'profit'],
            'lay1us_stake': ['lay1us', 'lay', '1us', 'stake', 'unit', 'bet', 'wager'],
            'lay1us_payout': ['lay1us', 'lay', '1us', 'payout', 'return', 'total', 'payment'],
            
            # Additional columns
            'odds': ['odds', 'price', 'decimal', 'odd'],
            'result': ['result', 'outcome', 'win', 'lose', 'push', 'status'],
            'date': ['date', 'match date', 'game date', 'day', 'time']
        }
        
        # Special handling for some common column name formats
        specialized_patterns = {
            # For AI percentages
            'ai_pct': [r'ai\s*\d*\s*%', r'ai.*percent', r'confidence', r'prob\s*%'],
            # For MDS columns
            'mds_stake': [r'mds.*stake', r'mds.*unit', r'md.*stake'],
            'mds_payout': [r'mds.*payout', r'mds.*return', r'md.*payout'],
            'mds_roi': [r'mds.*roi', r'mds.*profit', r'mds.*return.*investment'],
            # For 1US columns
            '1us_stake': [r'1us.*stake', r'1us.*unit', r'1u.*stake', r'one.*unit'],
            '1us_payout': [r'1us.*payout', r'1us.*return', r'1u.*payout'],
            '1us_roi': [r'1us.*roi', r'1us.*profit', r'1us.*return.*investment'],
            # For GOM/LAY columns
            'gom_stake': [r'gom.*stake', r'lay.*stake', r'gom.*unit'],
            'gom_payout': [r'gom.*payout', r'lay.*payout', r'gom.*return'],
            'gom_roi': [r'gom.*roi', r'lay.*roi', r'gom.*profit'],
        }
        
        # Try to find column names by exact match or case-insensitive substring
        for col in df.columns:
            col_str = str(col).lower()
            
            # First try exact column names (case-insensitive)
            for col_type in column_mapping.keys():
                if col_str == col_type.lower().replace('_', ' '):
                    column_mapping[col_type] = col
                    break
            
            # Then try partial matching with the patterns
            if column_mapping.get(col_type) is None:  # If not already found by exact match
                for col_type, patterns in column_patterns.items():
                    # Look for primary identifier followed by a secondary feature
                    primary_patterns = patterns[:1]  # First pattern is primary
                    secondary_patterns = patterns[1:]  # Rest are secondary
                    
                    # Check if column contains the primary pattern
                    primary_match = any(p.lower() in col_str for p in primary_patterns)
                    
                    # And at least one secondary pattern
                    secondary_match = any(p.lower() in col_str for p in secondary_patterns)
                    
                    if primary_match and (secondary_match or not secondary_patterns):
                        column_mapping[col_type] = col
                        break
        
        # If we haven't found important columns, try fuzzy matching using specialized patterns
        for col_type, regex_patterns in specialized_patterns.items():
            if column_mapping[col_type] is None:  # Only if not already found
                for col in df.columns:
                    col_str = str(col).lower()
                    for pattern in regex_patterns:
                        import re
                        if re.search(pattern, col_str):
                            column_mapping[col_type] = col
                            break
                    if column_mapping[col_type] is not None:
                        break
        
        # Last resort: Try to infer column meaning from data
        if column_mapping['ai_pct'] is None:
            # Look for percentage columns (values between 0-100 or 0-1)
            for col in df.columns:
                try:
                    # Check if column has numeric data
                    if pd.api.types.is_numeric_dtype(df[col]):
                        values = df[col].dropna()
                        if len(values) > 0:
                            # Check if values are in percent range
                            if ((values >= 0) & (values <= 1)).all() or ((values >= 0) & (values <= 100)).all():
                                column_mapping['ai_pct'] = col
                                logger.debug(f"Inferred AI percentage column from data: {col}")
                                break
                except Exception:
                    pass
        
        # Check data columns that might contain ROI info
        if column_mapping['mds_roi'] is None:
            for col in df.columns:
                try:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        values = df[col].dropna()
                        if len(values) > 0:
                            # ROI columns typically have positive and negative values around -1 to 1
                            if ((values >= -3) & (values <= 3)).mean() > 0.8:  # 80% of values in ROI range
                                column_mapping['mds_roi'] = col
                                logger.debug(f"Inferred MDS ROI column from data: {col}")
                                break
                except Exception:
                    pass
        
        # Special handling for specific layouts
        # If we see columns like "Model 3" or similar, these might be model names
        model_cols = [col for col in df.columns if 'model' in str(col).lower()]
        if model_cols and column_mapping['model_type'] is None:
            column_mapping['model_type'] = model_cols[0]
        
        # Log found and missing columns
        found_cols = [k for k, v in column_mapping.items() if v is not None]
        missing_cols = [k for k, v in column_mapping.items() if v is None]
        
        logger.info(f"Found columns: {found_cols}")
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
        
        # Handle the case where we're looking at a column explanation sheet instead of data
        if 'Column' in df.columns and 'Name' in df.columns and 'Explanation' in df.columns:
            logger.warning("This appears to be a column explanation sheet rather than actual data")
            
            # Try to map columns based on the explanation sheet
            try:
                for _, row in df.iterrows():
                    column_name = str(row.get('Name', '')).lower()
                    for col_type, patterns in column_patterns.items():
                        # If this explanation row matches our expected column type
                        if any(pattern.lower() in column_name for pattern in patterns):
                            # We found what the column name should be in the actual data
                            logger.info(f"Found column mapping from explanation: {column_name} -> {col_type}")
                            # But we don't have the actual data here - need to use this info for the real data file
            except Exception as e:
                logger.error(f"Error processing explanation sheet: {e}")
                
        return column_mapping#############################################
# BMXPattern Class
#############################################

class BMXPattern:
    """
    Represents a detected betting pattern with comprehensive performance metrics.
    """
    def __init__(self, 
                 model: str, 
                 league: str, 
                 selection: str, 
                 ai_min: float, 
                 ai_max: float, 
                 phases: Optional[List[int]] = None, 
                 staking_type: str = "MDS", 
                 is_weighted: bool = False,
                 model_type: str = "BACK"):
        """
        Initialize a betting pattern.
        
        Args:
            model (str): Name of the model.
            league (str): League name.
            selection (str): Betting selection.
            ai_min (float): Minimum AI percentage.
            ai_max (float): Maximum AI percentage.
            phases (List[int], optional): Phases included in the pattern.
            staking_type (str, optional): Staking strategy type.
            is_weighted (bool, optional): Whether season weighting is applied.
            model_type (str, optional): Type of model (BACK/LAY).
        """
        self.model = model
        self.league = league
        self.selection = selection
        self.ai_min = ai_min
        self.ai_max = ai_max
        self.phases = sorted(phases) if phases else list(range(1, 7))
        self.staking_type = staking_type
        self.is_weighted = is_weighted
        self.model_type = model_type
        
        # Performance metrics
        self.roi = None
        self.sample_size = None
        self.stake = None
        self.payout = None
        self.profit = None
        self.strike_rate = None
        
        # Season breakdown
        self.season_results = {}  # Dict mapping season number to performance metrics
        
        # Advanced analysis results
        self.phase_results = {}
        self.optimized_variants = []
        self.validation_metrics = {}
        self.trend_data = {}
        
        # Pattern detection metadata
        self.gaps = []  # List of tuples (gap_start, gap_end) representing gaps in the pattern
        self.contiguous_segments = []  # List of tuples (segment_start, segment_end)
    
    def generate_description(self) -> str:
        """
        Generate a comprehensive description of the pattern.
        
        Returns:
            str: Detailed pattern description.
        """
        phases_str = "-".join(map(str, self.phases)) if self.phases else "All"
        weight_str = " (Weighted)" if self.is_weighted else ""
        roi_str = f"{self.roi*100:.2f}%" if self.roi is not None else "N/A"
        
        return (f"{self.model} [{self.model_type}]: {self.league} {self.selection} - "
                f"AI {self.ai_min:.1f}%-{self.ai_max:.1f}% - "
                f"Phases {phases_str} - {self.staking_type}{weight_str} - "
                f"ROI: {roi_str} (n={self.sample_size or 0})")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert pattern to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the pattern.
        """
        confidence_score = self.validation_metrics.get('confidence_score', 0)
        
        return {
            'Model': self.model,
            'Model_Type': self.model_type,
            'League': self.league,
            'Selection': self.selection,
            'AI_Min': self.ai_min,
            'AI_Max': self.ai_max,
            'Phases': "-".join(map(str, self.phases)) if self.phases else "All",
            'Staking_Type': self.staking_type,
            'ROI': self.roi,
            'Sample_Size': self.sample_size,
            'Total_Stake': self.stake,
            'Total_Payout': self.payout,
            'Total_Profit': self.profit,
            'Strike_Rate': self.strike_rate,
            'Confidence_Score': confidence_score,
            'Profitable_Season_Ratio': self.validation_metrics.get('profitable_season_ratio', 0),
            'ROI_Stability': self.validation_metrics.get('roi_stability', 0),
            'Statistical_Significance': self.validation_metrics.get('statistical_significance', 1.0),
            'Bootstrap_Mean': self.validation_metrics.get('bootstrap_mean', 0),
            'Bootstrap_CI_Lower': self.validation_metrics.get('bootstrap_ci_lower', 0),
            'Bootstrap_CI_Upper': self.validation_metrics.get('bootstrap_ci_upper', 0),
            'Trend_Direction': self.trend_data.get('trend_direction', 'Stable'),
            'Sharpe_Ratio': self.validation_metrics.get('sharpe_ratio', 0),
            'Pattern_Has_Gaps': len(self.gaps) > 0,
            'Number_Of_Gaps': len(self.gaps),
            'Is_Weighted': self.is_weighted
        }


#############################################
# BMXAnalyzer Class
#############################################

class BMXAnalyzer:
    """
    Core analysis engine for detecting and validating betting patterns.
    """
    def __init__(self, config: BMXConfig):
        """
        Initialize the analyzer with configuration.
        
        Args:
            config (BMXConfig): Configuration object.
        """
        self.config = config
        logger.info("BMX Analyzer initialized")
    
    def detect_patterns(self, df: pd.DataFrame, model_name: str = "Detected Model") -> List[BMXPattern]:
        """
        Detect betting patterns in the input DataFrame.
        
        Args:
            df (pd.DataFrame): Input data.
            model_name (str): Name to assign to detected patterns.
        
        Returns:
            List[BMXPattern]: Detected patterns.
        """
        logger.info(f"Starting pattern detection for {model_name}")
        
        # Preprocessing
        df = self.config.normalize_ai_percentages(df)
        column_mapping = self.config.identify_columns(df)
        
        # Check for required columns
        required_cols = ['league', 'selection', 'ai_pct', 'phase', 'season_number', 'model_type']
        missing_cols = [col for col in required_cols if column_mapping.get(col) is None]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            return []
        
        # Get unique league-selection combinations
        unique_combinations = self._get_unique_combinations(df, column_mapping)
        logger.info(f"Found {len(unique_combinations)} unique league-selection combinations")
        
        # Get unique model types
        model_type_col = column_mapping.get('model_type')
        model_types = df[model_type_col].unique() if model_type_col else ["BACK"]
        
        all_patterns = []
        
        # Process each league-selection-model_type combination
        for league, selection in unique_combinations:
            for model_type in model_types:
                # Determine if this is a BACK or LAY model
                if model_type_col:
                    model_type_value = model_type
                    is_lay_model = any(lay_keyword in str(model_type).upper() for lay_keyword in ['LAY', 'L'])
                    model_type_str = "LAY" if is_lay_model else "BACK"
                else:
                    # Default to BACK if no model type column
                    model_type_value = None
                    model_type_str = "BACK"
                
                # Filter data for this combination
                filtered_data = self._filter_data(df, column_mapping, league, selection, model_type_value)
                
                if len(filtered_data) < self.config.min_sample_size:
                    logger.debug(f"Insufficient data for {league} {selection} {model_type_str}: {len(filtered_data)} rows")
                    continue
                
                # Process with different staking types based on model type
                for roi_direction in ["positive", "negative"]:
                    staking_types = self.config.staking_types.get(model_type_str, {}).get(roi_direction, [])
                    
                    for staking_type in staking_types:
                        stake_col = column_mapping.get(f'{staking_type.lower()}_stake')
                        payout_col = column_mapping.get(f'{staking_type.lower()}_payout')
                        
                        if not stake_col or not payout_col:
                            logger.warning(f"Missing columns for staking type {staking_type}: {stake_col=}, {payout_col=}")
                            continue
                        
                        # Generate heatmap
                        heatmap = self._generate_ai_band_heatmap(
                            filtered_data, 
                            column_mapping, 
                            league, 
                            selection, 
                            staking_type,
                            model_type_str
                        )
                        
                        if heatmap.empty:
                            logger.debug(f"No valid heatmap data for {league} {selection} {model_type_str} {staking_type}")
                            continue
                        
                        # Extract patterns from heatmap
                        is_positive = roi_direction == "positive"
                        patterns = self._extract_patterns_from_heatmap(
                            heatmap, 
                            model_name,
                            league, 
                            selection, 
                            staking_type,
                            model_type_str,
                            is_positive
                        )
                        
                        if patterns:
                            logger.info(f"Found {len(patterns)} {roi_direction} patterns for {league} {selection} {model_type_str} {staking_type}")
                            all_patterns.extend(patterns)
        
        logger.info(f"Pattern detection complete. Found {len(all_patterns)} total patterns")
        return all_patternsdef _get_unique_combinations(self, df: pd.DataFrame, column_mapping: Dict[str, Optional[str]]) -> List[Tuple[str, str]]:
        """
        Get unique league and selection combinations from the DataFrame.
        
        Args:
            df (pd.DataFrame): Input data.
            column_mapping (Dict[str, Optional[str]]): Column mapping.
        
        Returns:
            List[Tuple[str, str]]: List of (league, selection) tuples.
        """
        league_col = column_mapping.get('league')
        selection_col = column_mapping.get('selection')
        
        if not league_col or not selection_col:
            logger.error(f"Missing league or selection columns: {league_col=}, {selection_col=}")
            return []
        
        combinations = df[[league_col, selection_col]].drop_duplicates().values.tolist()
        return [(str(league), str(selection)) for league, selection in combinations]
    
    def _filter_data(self, df: pd.DataFrame, column_mapping: Dict[str, Optional[str]], 
                    league: str, selection: str, model_type: Optional[Any] = None) -> pd.DataFrame:
        """
        Filter DataFrame for a specific league, selection, and model type.
        
        Args:
            df (pd.DataFrame): Input data.
            column_mapping (Dict[str, Optional[str]]): Column mapping.
            league (str): League to filter by.
            selection (str): Selection to filter by.
            model_type (Any, optional): Model type to filter by.
        
        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        league_col = column_mapping.get('league')
        selection_col = column_mapping.get('selection')
        model_type_col = column_mapping.get('model_type')
        
        if not league_col or not selection_col:
            return pd.DataFrame()
        
        # Start with league and selection filter
        mask = (df[league_col].astype(str) == league) & (df[selection_col].astype(str) == selection)
        
        # Add model type filter if specified and the column exists
        if model_type is not None and model_type_col:
            mask = mask & (df[model_type_col] == model_type)
        
        return df[mask].copy()
    
    def _generate_ai_band_heatmap(self, df: pd.DataFrame, column_mapping: Dict[str, Optional[str]], 
                                 league: str, selection: str, staking_type: str, model_type: str) -> pd.DataFrame:
        """
        Generate AI percentage band heatmap for ROI analysis.
        
        Args:
            df (pd.DataFrame): Filtered data for specific league and selection.
            column_mapping (Dict[str, Optional[str]]): Column mapping.
            league (str): League name.
            selection (str): Selection name.
            staking_type (str): Staking type (e.g., "MDS", "1US").
            model_type (str): Model type ("BACK" or "LAY").
        
        Returns:
            pd.DataFrame: Heatmap data with AI bands and ROI values.
        """
        ai_pct_col = column_mapping.get('ai_pct')
        stake_col = column_mapping.get(f'{staking_type.lower()}_stake')
        payout_col = column_mapping.get(f'{staking_type.lower()}_payout')
        phase_col = column_mapping.get('phase')
        season_col = column_mapping.get('season_number')
        
        if not ai_pct_col or not stake_col or not payout_col:
            logger.warning(f"Missing required columns for heatmap: {ai_pct_col=}, {stake_col=}, {payout_col=}")
            return pd.DataFrame()
        
        # Process through each AI percentage band (1% increments)
        heatmap_results = []
        
        for ai_min in range(0, 100):
            ai_max = ai_min + 1
            
            # Filter data for this AI band
            band_data = df[(df[ai_pct_col] >= ai_min) & (df[ai_pct_col] < ai_max)]
            
            if len(band_data) < 1:
                continue
            
            # Calculate ROI for this band
            total_stake = band_data[stake_col].sum()
            total_payout = band_data[payout_col].sum()
            
            if total_stake <= 0:
                continue
            
            roi = (total_payout / total_stake) - 1
            profit = total_payout - total_stake
            
            # Calculate strike rate (percentage of winning bets)
            wins = band_data[band_data[payout_col] > band_data[stake_col]].shape[0]
            strike_rate = wins / len(band_data) if len(band_data) > 0 else 0
            
            # Get phase distribution if available
            phase_distribution = {}
            if phase_col:
                for phase in sorted(band_data[phase_col].unique()):
                    phase_data = band_data[band_data[phase_col] == phase]
                    if len(phase_data) >= self.config.min_sample_size_by_phase:
                        phase_stake = phase_data[stake_col].sum()
                        phase_payout = phase_data[payout_col].sum()
                        phase_roi = (phase_payout / phase_stake) - 1 if phase_stake > 0 else None
                        if phase_roi is not None:
                            phase_distribution[int(phase)] = {
                                'sample_size': len(phase_data),
                                'roi': phase_roi,
                                'stake': phase_stake,
                                'payout': phase_payout
                            }
            
            # Get season distribution if available
            season_distribution = {}
            if season_col:
                for season in sorted(band_data[season_col].unique()):
                    season_data = band_data[band_data[season_col] == season]
                    season_stake = season_data[stake_col].sum()
                    season_payout = season_data[payout_col].sum()
                    season_roi = (season_payout / season_stake) - 1 if season_stake > 0 else None
                    if season_roi is not None:
                        season_distribution[str(season)] = {
                            'sample_size': len(season_data),
                            'roi': season_roi,
                            'stake': season_stake,
                            'payout': season_payout
                        }
            
            heatmap_results.append({
                'League': league,
                'Selection': selection,
                'Staking_Type': staking_type,
                'Model_Type': model_type,
                'AI_Min': ai_min,
                'AI_Max': ai_max,
                'ROI': roi,
                'Profit': profit,
                'Sample_Size': len(band_data),
                'Total_Stake': total_stake,
                'Total_Payout': total_payout,
                'Strike_Rate': strike_rate,
                'Phase_Distribution': phase_distribution,
                'Season_Distribution': season_distribution
            })
        
        return pd.DataFrame(heatmap_results)
    
    def _extract_patterns_from_heatmap(self, heatmap: pd.DataFrame, model_name: str,
                                      league: str, selection: str, staking_type: str,
                                      model_type: str, is_positive: bool = True) -> List[BMXPattern]:
        """
        Extract patterns from the AI band heatmap with flexible gap tolerance.
        
        Args:
            heatmap (pd.DataFrame): Heatmap data.
            model_name (str): Name of the model.
            league (str): League name.
            selection (str): Selection name.
            staking_type (str): Staking type.
            model_type (str): Model type ("BACK" or "LAY").
            is_positive (bool): Whether to look for positive or negative patterns.
        
        Returns:
            List[BMXPattern]: Detected patterns.
        """
        # Filter data based on ROI direction and sample size
        if is_positive:
            valid_data = heatmap[(heatmap['ROI'] > 0) & 
                                (heatmap['Sample_Size'] >= self.config.min_sample_size)]
            roi_threshold = self.config.min_pos_roi
        else:
            valid_data = heatmap[(heatmap['ROI'] < 0) & 
                                (heatmap['Sample_Size'] >= self.config.min_sample_size)]
            roi_threshold = self.config.min_neg_roi
        
        if valid_data.empty:
            return []
        
        # Sort by AI percentage for sequential analysis
        sorted_data = valid_data.sort_values('AI_Min')
        
        # Initialize pattern detection
        patterns = []
        used_ai_bands = set()
        
        for start_idx in range(len(sorted_data)):
            if start_idx in used_ai_bands:
                continue
                
            # Initialize a new potential pattern
            start_band = sorted_data.iloc[start_idx]
            current_pattern = [start_band]
            gaps = []
            gap_start = None
            
            # Look ahead for consecutive bands and tolerated gaps
            for next_idx in range(start_idx + 1, len(sorted_data)):
                next_band = sorted_data.iloc[next_idx]
                prev_band = current_pattern[-1]
                
                # Check if this band is adjacent to the previous one
                is_adjacent = next_band['AI_Min'] == prev_band['AI_Max']
                
                # Calculate the gap size if not adjacent
                gap_size = next_band['AI_Min'] - prev_band['AI_Max'] if not is_adjacent else 0
                
                # Allow adding this band if:
                # 1. It's adjacent to the previous band, OR
                # 2. The gap is within allowed size AND we haven't exceeded max gaps
                if is_adjacent:
                    # If we were tracking a gap, close it
                    if gap_start is not None:
                        gaps.append((gap_start, prev_band['AI_Max']))
                        gap_start = None
                    current_pattern.append(next_band)
                    
                elif (gap_size <= self.config.max_pattern_gaps and 
                      len(gaps) < self.config.max_pattern_gaps):
                    # Start tracking a new gap if not already tracking one
                    if gap_start is None:
                        gap_start = prev_band['AI_Max']
                    
                    # Check if adding this band would maintain acceptable ROI
                    # Calculate combined ROI including all bands so far plus this new one
                    combined_stake = sum(band['Total_Stake'] for band in current_pattern) + next_band['Total_Stake']
                    combined_payout = sum(band['Total_Payout'] for band in current_pattern) + next_band['Total_Payout']
                    combined_roi = (combined_payout / combined_stake) - 1 if combined_stake > 0 else None
                    
                    # Only add this band if it doesn't hurt the pattern's ROI too much
                    roi_drop = abs((current_pattern[-1]['ROI'] - combined_roi) if combined_roi is not None else float('inf'))
                    
                    if ((is_positive and combined_roi is not None and combined_roi > roi_threshold) or
                        (not is_positive and combined_roi is not None and combined_roi < roi_threshold)):
                        if roi_drop <= self.config.pattern_gap_tolerance:
                            current_pattern.append(next_band)
                        else:
                            # This band hurts ROI too much, stop expanding
                            break
                    else:
                        # ROI would fall below threshold, stop expanding
                        break
                else:
                    # Gap too large or too many gaps, stop expanding
                    break
