import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreparer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, df):
        """
        Prepare data for training:
        1. Handle missing values
        2. Scale the data
        3. Return both scaled data and processed dataframe
        
        Args:
            df: Input DataFrame with stock data
            
        Returns:
            tuple: (scaled_data as numpy array, processed dataframe)
        """
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Handle missing values
        df_processed = df_processed.fillna(method='ffill')
        if df_processed.isnull().any().any():
            df_processed = df_processed.fillna(method='bfill')
            
        # Drop any remaining rows with NaN if any still exist
        df_processed = df_processed.dropna()
        
        # Assuming the first column is the date, drop it for scaling
        if 'Date' in df_processed.columns:
            dates = df_processed['Date']
            df_processed = df_processed.drop('Date', axis=1)
            
        # Scale the data
        scaled_data = self.scaler.fit_transform(df_processed)
        
        return scaled_data, df_processed