import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreparer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def prepare_data(self, df, fit_scaler=True):
        """
        Prepare data for training WITHOUT future information leakage:
        1. Handle missing values
        2. Scale the data using only training data statistics
        3. Return both scaled data and processed dataframe
        
        Args:
            df: Input DataFrame with stock data
            fit_scaler: Whether to fit the scaler (True for train, False for val/test)
            
        Returns:
            tuple: (scaled_data as numpy array, processed dataframe)
        """
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Handle missing values - forward fill then backward fill
        df_processed = df_processed.fillna(method='ffill')
        if df_processed.isnull().any().any():
            df_processed = df_processed.fillna(method='bfill')
            
        # Drop any remaining rows with NaN if any still exist
        df_processed = df_processed.dropna()
        
        # Remove date column if present
        if 'Date' in df_processed.columns:
            df_processed = df_processed.drop('Date', axis=1)
            
        # CRITICAL FIX: Only fit scaler on training data
        if fit_scaler and not self.is_fitted:
            scaled_data = self.scaler.fit_transform(df_processed)
            self.is_fitted = True
        elif self.is_fitted:
            # Use the already fitted scaler for validation/test data
            scaled_data = self.scaler.transform(df_processed)
        else:
            raise ValueError("Scaler must be fitted on training data first")
        
        return scaled_data, df_processed
    
    def prepare_temporal_splits(self, df, train_ratio=0.8, val_ratio=0.1):
        """
        Create proper temporal splits with correct scaling
        
        Args:
            df: Full DataFrame
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            
        Returns:
            tuple: (train_scaled, val_scaled, test_scaled, train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Split dataframes temporally
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        # Scale data properly - fit only on training data
        train_scaled, train_processed = self.prepare_data(train_df, fit_scaler=True)
        val_scaled, val_processed = self.prepare_data(val_df, fit_scaler=False)
        test_scaled, test_processed = self.prepare_data(test_df, fit_scaler=False)
        
        return (train_scaled, val_scaled, test_scaled, 
                train_processed, val_processed, test_processed)
    
    def inverse_transform(self, scaled_predictions, target_column_idx=0):
        """
        Convert scaled predictions back to original scale
        
        Args:
            scaled_predictions: Predictions in scaled format
            target_column_idx: Index of the target column in the original data
            
        Returns:
            numpy array: Predictions in original scale
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
            
        # Create dummy array with same shape as training data
        n_features = len(self.scaler.scale_)
        dummy = np.zeros((len(scaled_predictions), n_features))
        dummy[:, target_column_idx] = scaled_predictions.flatten()
        
        # Inverse transform and extract target column
        inverse_dummy = self.scaler.inverse_transform(dummy)
        return inverse_dummy[:, target_column_idx]
    
    def reset_scaler(self):
        """Reset the scaler for new data"""
        self.scaler = MinMaxScaler()
        self.is_fitted = False
