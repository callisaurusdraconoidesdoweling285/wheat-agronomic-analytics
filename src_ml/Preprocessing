"""
Agronomic Tabular Data Preprocessing Pipeline

This script cleans, imputes, and aggregates plot-level wheat yield data 
and daily weather data into a single, machine-learning-ready dataset. 
It preserves metadata keys (like TrialCode) to allow for future integration 
with computer vision-derived phenotypic traits.
"""

import pandas as pd
import numpy as np
import os
import warnings

# Suppress warnings for cleaner execution output
warnings.filterwarnings('ignore')

def load_and_standardize_data(plot_path: str, weather_path: str):
    """Loads CSV files and standardizes primary keys."""
    print("Loading datasets...")
    plot_df = pd.read_csv(plot_path, parse_dates=['SowingDate', 'HarvestDate'])
    weather_df = pd.read_csv(weather_path, parse_dates=['Date'])

    # Standardize TrialCode to numeric only (e.g., 'trial_1380' -> 1380)
    # This acts as the primary key for joining ML and CV data later
    plot_df['TrialCode'] = plot_df['TrialCode'].astype(str).str.extract(r'(\d+)', expand=False).astype(float)
    weather_df['TrialCode'] = weather_df['TrialCode'].astype(str).str.extract(r'(\d+)', expand=False).astype(float)
    
    return plot_df, weather_df

def clean_and_impute_plots(plot_df: pd.DataFrame, missing_threshold: float = 0.30) -> pd.DataFrame:
    """Drops sparse columns and imputes missing numeric values."""
    print(f"Dropping columns with >{missing_threshold*100}% missing values...")
    # Drop columns with too many missing values
    plot_df = plot_df[plot_df.columns[plot_df.isnull().mean() <= missing_threshold]]

    print("Imputing missing numeric values...")
    # Impute missing numeric features using the median of that specific TrialCode
    # If the entire TrialCode is missing that feature, fallback to the global median
    numeric_cols = plot_df.select_dtypes(include='number').columns
    for col in numeric_cols:
        plot_df[col] = plot_df.groupby('TrialCode')[col].transform(lambda x: x.fillna(x.median()))
        plot_df[col].fillna(plot_df[col].median(), inplace=True)
        
    return plot_df

def aggregate_weather_features(plot_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates daily weather data into seasonal features per trial."""
    print("Aggregating weather data across growing seasons...")
    weather_feats = []
    
    for trial, grp in plot_df.groupby('TrialCode'):
        # Define the growing season window for this trial
        sow_date = grp['SowingDate'].min()
        harv_date = grp['HarvestDate'].max()
        
        # Filter weather data for this specific trial and timeframe
        sub = weather_df[(weather_df['TrialCode'] == trial) &
                         (weather_df['Date'] >= sow_date) &
                         (weather_df['Date'] <= harv_date)]
        
        if sub.empty:
            continue
            
        # Engineer seasonal weather features
        weather_feats.append({
            'TrialCode': trial,
            'TotalRain': sub['Rain'].sum(),
            'AvgTmax':  sub['T.Max'].mean(),
            'AvgTmin':  sub['T.Min'].mean(),
            'TotalEvap': sub['Evap'].sum(),
            'TotalRadn': sub['Radn'].sum(),
            'AvgVP':    sub['VP'].mean(),
            'AvgRHmaxT':sub['RHmaxT'].mean(),
            'AvgRHminT':sub['RHminT'].mean()
        })
        
    return pd.DataFrame(weather_feats)

def remove_multicollinearity(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Removes highly correlated numeric features to prevent data leakage/overfitting."""
    print(f"Removing features with correlation > {threshold}...")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Dropped highly correlated features: {to_drop}")
    
    return df.drop(columns=to_drop)

def main():
    # File paths (Adjust these to match your repository structure)
    PLOT_DATA_PATH = 'data/combined_plot_data.csv'
    WEATHER_DATA_PATH = 'data/combined_all_weather.csv'
    OUTPUT_PATH = 'data/ml_ready_aggregated_data.csv'
    
    # 1. Load Data
    try:
        plot_df, weather_df = load_and_standardize_data(PLOT_DATA_PATH, WEATHER_DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure data files are in the correct directory.")
        return

    # 2. Clean and Impute Plot Data
    plot_df_cleaned = clean_and_impute_plots(plot_df)

    # 3. Aggregate Weather Data
    weather_agg_df = aggregate_weather_features(plot_df_cleaned, weather_df)

    # 4. Merge Plot and Weather Data
    print("Merging plot and weather pipelines...")
    merged_df = pd.merge(plot_df_cleaned, weather_agg_df, on='TrialCode', how='left')
    
    # Drop any trials that didn't have matching weather data
    merged_df.dropna(subset=['TotalRain'], inplace=True) 

    # 5. Feature Selection (Drop Multicollinear features)
    final_df = remove_multicollinearity(merged_df)

    # 6. Save final output
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Preprocessing complete! Cleaned data saved to {OUTPUT_PATH}")
    print(f"Final dataset shape: {final_df.shape}")

if __name__ == "__main__":
    main()
