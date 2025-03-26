import gc
import os
import psutil
import pandas as pd
import numpy as np
import warnings
import logging
from sklearn.cluster import KMeans
import math

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Class to monitor and report memory usage"""
    
    @staticmethod
    def get_usage_mb():
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def log_usage(label):
        """Log current memory usage with a label"""
        usage = MemoryMonitor.get_usage_mb()
        logger.info(f"Memory usage at {label}: {usage:.2f} MB")
        return usage
    
    @staticmethod
    def force_release():
        """Force garbage collection"""
        gc.collect()
        return MemoryMonitor.get_usage_mb()

class DataSampler:
    """Class providing various data sampling techniques for memory optimization"""
    
    @staticmethod
    def random_sample(df, max_points=5000):
        """Simple random sampling"""
        if len(df) <= max_points:
            return df
        return df.sample(max_points, random_state=42)
    
    @staticmethod
    def stratified_sample(df, column, max_points=5000):
        """Stratified sampling to maintain distribution in a categorical column"""
        if len(df) <= max_points or column not in df.columns:
            return df
            
        result = pd.DataFrame()
        for category in df[column].unique():
            category_df = df[df[column] == category]
            category_size = max(1, int(max_points * len(category_df) / len(df)))
            result = pd.concat([result, category_df.sample(min(category_size, len(category_df)), random_state=42)])
        return result
    
    @staticmethod
    def cluster_sample(df, max_points=5000, numeric_cols=None, n_clusters=None):
        """Cluster-based sampling to maintain data structure"""
        if len(df) <= max_points:
            return df
            
        # Choose numeric columns for clustering
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return DataSampler.random_sample(df, max_points)
            if len(numeric_cols) > 5:
                numeric_cols = numeric_cols[:5]
        
        # Handle missing values
        cluster_data = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = min(int(math.sqrt(max_points)), 50)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df_copy = df.copy()
                df_copy['cluster'] = kmeans.fit_predict(cluster_data)
                
                # Sample from each cluster
                result = pd.DataFrame()
                for cluster in range(n_clusters):
                    cluster_df = df_copy[df_copy['cluster'] == cluster]
                    cluster_size = max(1, int(max_points * len(cluster_df) / len(df)))
                    result = pd.concat([result, cluster_df.sample(min(cluster_size, len(cluster_df)), random_state=42)])
                
                return result.drop(columns=['cluster'])
        except Exception as e:
            logger.warning(f"Cluster sampling failed: {str(e)}")
            return DataSampler.random_sample(df, max_points)
    
    @staticmethod
    def smart_sample(df, max_points=5000, method='auto'):
        """Intelligently choose a sampling method based on the data"""
        if df is None or df.empty or len(df) <= max_points:
            return df
            
        if method == 'auto':
            # Choose method based on data characteristics
            if 'Risk_Category' in df.columns:
                return DataSampler.stratified_sample(df, 'Risk_Category', max_points)
            elif len(df.select_dtypes(include=['number']).columns) >= 2:
                return DataSampler.cluster_sample(df, max_points)
            else:
                return DataSampler.random_sample(df, max_points)
        elif method == 'random':
            return DataSampler.random_sample(df, max_points)
        elif method == 'stratified' and 'Risk_Category' in df.columns:
            return DataSampler.stratified_sample(df, 'Risk_Category', max_points)
        elif method == 'cluster':
            return DataSampler.cluster_sample(df, max_points)
        else:
            return DataSampler.random_sample(df, max_points)

# Example usage
if __name__ == "__main__":
    # Test with a sample dataframe
    df = pd.DataFrame({
        'A': np.random.rand(10000),
        'B': np.random.rand(10000),
        'C': np.random.choice(['X', 'Y', 'Z'], 10000)
    })
    
    MemoryMonitor.log_usage("Original")
    sampled = DataSampler.smart_sample(df, 1000)
    MemoryMonitor.log_usage("After sampling")
    print(f"Original size: {len(df)}, Sampled size: {len(sampled)}")
