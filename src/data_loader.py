"""
Data Loading and Cleaning Module

Handles loading delivery data from CSV files and performing data validation and cleaning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Loads and cleans delivery data from CSV files.

    Performs validation and cleaning operations including:
    - Required column validation
    - Null coordinate removal
    - Duplicate AWB removal
    - Coordinate bounds validation (Indonesia)
    - Data quality reporting
    """

    # Indonesia coordinate bounds
    LATITUDE_MIN = -11
    LATITUDE_MAX = 6
    LONGITUDE_MIN = 95
    LONGITUDE_MAX = 141

    # Required columns for the system
    REQUIRED_COLUMNS = [
        'AWB_NUMBER',
        'EMPLOYEE_ID',
        'NICKNAME',
        'DO_POD_DELIVER_CODE',
        'BERATASLI',
        'SELECTED_LATITUDE',
        'SELECTED_LONGITUDE',
        'BRANCH_LATITUDE',
        'BRANCH_LONGITUDE'
    ]

    # Optional columns with alternative names
    OPTIONAL_COLUMNS = {
        'BRANCH_NAME': ['GERAI', 'BRANCH_NAME'],
        'BRANCH_CODE': ['KODE_GERAI', 'BRANCH_CODE'],
        'DELIVERY_DATE': ['DO_POD_DELIVER_DATE', 'DELIVERY_DATE']
    }

    def __init__(self):
        """Initialize DataLoader"""
        self.cleaning_stats = {}

    def load_and_clean(self, csv_path: str, output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Load, clean, and validate delivery data from CSV file.

        Args:
            csv_path: Path to the input CSV file
            output_dir: Optional directory to save cleaned data

        Returns:
            Cleaned pandas DataFrame

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
        """
        print("\n" + "="*80)
        print("DATA LOADING AND CLEANING")
        print("="*80)

        # Check if file exists
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        print(f"\n📂 Loading data from: {csv_path}")

        # Load data
        df = pd.read_csv(csv_path)
        initial_count = len(df)
        print(f"  • Initial rows: {initial_count:,}")

        # Validate required columns and map alternative names
        df = self._validate_columns(df)

        # Cleaning steps
        df = self._remove_null_coordinates(df)
        df = self._remove_duplicates(df)
        df = self._validate_coordinates(df)
        df = self._add_delivery_id(df)

        # Calculate cleaning statistics
        final_count = len(df)
        self.cleaning_stats = {
            'initial_rows': initial_count,
            'final_rows': final_count,
            'rows_removed': initial_count - final_count,
            'removal_percentage': ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0,
            'unique_awbs': df['AWB_NUMBER'].nunique(),
            'unique_couriers': df['EMPLOYEE_ID'].nunique(),
            'unique_pods': df['DO_POD_DELIVER_CODE'].nunique(),
            'total_weight_kg': df['BERATASLI'].sum(),
            'date_range': (df['DELIVERY_DATE'].min(), df['DELIVERY_DATE'].max()) if 'DELIVERY_DATE' in df.columns else None
        }

        # Print summary
        self._print_cleaning_summary()

        # Save cleaned data if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"{Path(csv_path).stem}_clean.csv"
            df.to_csv(output_file, index=False)
            print(f"\n💾 Cleaned data saved to: {output_file}")

        return df

    def _validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that all required columns exist in the dataframe.
        Maps alternative column names to standard names.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with standardized column names

        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}\n"
                f"Available columns: {list(df.columns)}"
            )

        # Map optional columns to standard names
        for standard_name, alternatives in self.OPTIONAL_COLUMNS.items():
            for alt_name in alternatives:
                if alt_name in df.columns and standard_name not in df.columns:
                    df[standard_name] = df[alt_name]
                    print(f"  ℹ️  Mapped '{alt_name}' → '{standard_name}'")
                    break

        print(f"  ✓ All required columns present ({len(self.REQUIRED_COLUMNS)} columns)")

        return df

    def _remove_null_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with null delivery or branch coordinates.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with null coordinates removed
        """
        initial_count = len(df)

        # Check for null values in critical coordinate columns
        null_delivery_lat = df['SELECTED_LATITUDE'].isnull().sum()
        null_delivery_lon = df['SELECTED_LONGITUDE'].isnull().sum()
        null_branch_lat = df['BRANCH_LATITUDE'].isnull().sum()
        null_branch_lon = df['BRANCH_LONGITUDE'].isnull().sum()

        # Remove rows with null coordinates
        df = df.dropna(subset=[
            'SELECTED_LATITUDE',
            'SELECTED_LONGITUDE',
            'BRANCH_LATITUDE',
            'BRANCH_LONGITUDE'
        ])

        removed_count = initial_count - len(df)

        if removed_count > 0:
            print(f"  ⚠️  Removed {removed_count} rows with null coordinates:")
            print(f"      • Null delivery latitude: {null_delivery_lat}")
            print(f"      • Null delivery longitude: {null_delivery_lon}")
            print(f"      • Null branch latitude: {null_branch_lat}")
            print(f"      • Null branch longitude: {null_branch_lon}")
        else:
            print(f"  ✓ No null coordinates found")

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate AWB numbers, keeping the first occurrence.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)

        # Remove duplicates based on AWB_NUMBER
        df = df.drop_duplicates(subset='AWB_NUMBER', keep='first')

        removed_count = initial_count - len(df)

        if removed_count > 0:
            print(f"  ⚠️  Removed {removed_count} duplicate AWB numbers (kept first occurrence)")
        else:
            print(f"  ✓ No duplicate AWB numbers found")

        return df

    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that coordinates are within Indonesia bounds.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with invalid coordinates removed
        """
        initial_count = len(df)

        # Filter delivery coordinates
        df = df[
            (df['SELECTED_LATITUDE'].between(self.LATITUDE_MIN, self.LATITUDE_MAX)) &
            (df['SELECTED_LONGITUDE'].between(self.LONGITUDE_MIN, self.LONGITUDE_MAX))
        ]

        # Filter branch coordinates
        df = df[
            (df['BRANCH_LATITUDE'].between(self.LATITUDE_MIN, self.LATITUDE_MAX)) &
            (df['BRANCH_LONGITUDE'].between(self.LONGITUDE_MIN, self.LONGITUDE_MAX))
        ]

        removed_count = initial_count - len(df)

        if removed_count > 0:
            print(f"  ⚠️  Removed {removed_count} rows with coordinates outside Indonesia bounds")
            print(f"      • Valid latitude range: {self.LATITUDE_MIN} to {self.LATITUDE_MAX}")
            print(f"      • Valid longitude range: {self.LONGITUDE_MIN} to {self.LONGITUDE_MAX}")
        else:
            print(f"  ✓ All coordinates within Indonesia bounds")

        return df

    def _add_delivery_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sequential delivery_id column for internal tracking.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with delivery_id column added
        """
        df = df.reset_index(drop=True)
        df['delivery_id'] = range(len(df))
        print(f"  ✓ Added delivery_id column (0 to {len(df)-1})")

        return df

    def _print_cleaning_summary(self) -> None:
        """Print summary of data cleaning results."""
        print("\n" + "-"*80)
        print("CLEANING SUMMARY")
        print("-"*80)
        print(f"  Initial rows:          {self.cleaning_stats['initial_rows']:,}")
        print(f"  Final rows:            {self.cleaning_stats['final_rows']:,}")
        print(f"  Rows removed:          {self.cleaning_stats['rows_removed']:,} "
              f"({self.cleaning_stats['removal_percentage']:.1f}%)")
        print(f"\n  Data Quality:")
        print(f"  • Unique AWB numbers:  {self.cleaning_stats['unique_awbs']:,}")
        print(f"  • Unique couriers:     {self.cleaning_stats['unique_couriers']:,}")
        print(f"  • Unique PODs:         {self.cleaning_stats['unique_pods']:,}")
        print(f"  • Total weight:        {self.cleaning_stats['total_weight_kg']:,.1f} kg")

        if self.cleaning_stats['date_range']:
            print(f"  • Date range:          {self.cleaning_stats['date_range'][0]} to "
                  f"{self.cleaning_stats['date_range'][1]}")

        print("-"*80)

    def get_cleaning_stats(self) -> Dict:
        """
        Get dictionary of cleaning statistics.

        Returns:
            Dictionary containing cleaning statistics
        """
        return self.cleaning_stats


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    df = loader.load_and_clean(
        'data/raw/sleman_depok_10_10.csv',
        output_dir='data/processed'
    )
    print(f"\n✅ Successfully loaded and cleaned {len(df)} deliveries")
