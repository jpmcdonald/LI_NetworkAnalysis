import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from datetime import datetime
import re
import json
import os
import matplotlib.pyplot as plt
import csv
import hashlib
import uuid

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_cleaning.log', mode='w')  # Changed to 'w' mode to start fresh each run
    ]
)
logger = logging.getLogger(__name__)

def log_dataset_stats(df: pd.DataFrame, stage: str, dataset_name: str) -> None:
    """Log detailed statistics about the dataset at different stages."""
    logger.info(f"\n{'='*50}")
    logger.info(f"{dataset_name} Dataset Statistics - {stage}")
    logger.info(f"{'='*50}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Log detailed column statistics
    logger.info("\nColumn Statistics:")
    for column in df.columns:
        logger.info(f"\n{column}:")
        # Total non-null values
        non_null = df[column].notna().sum()
        logger.info(f"  Total non-null values: {non_null:,}")
        
        # For numeric columns
        if df[column].dtype in ['int64', 'float64']:
            total = df[column].sum()
            mean = df[column].mean()
            logger.info(f"  Sum: {total:,.2f}")
            logger.info(f"  Mean: {mean:,.2f}")
        
        # For datetime columns
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date = df[column].min()
            max_date = df[column].max()
            logger.info(f"  Date range: {min_date} to {max_date}")
        
        # For text/categorical columns
        else:
            # Count unique values
            unique_count = df[column].nunique()
            logger.info(f"  Unique values: {unique_count:,}")
            
            # Show top 5 most common values and their counts
            if unique_count > 0:
                value_counts = df[column].value_counts().head(5)
                logger.info("  Top 5 most common values:")
                for value, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    logger.info(f"    {value}: {count:,} ({percentage:.1f}%)")
        
        # Missing values
        missing = df[column].isna().sum()
        if missing > 0:
            missing_pct = (missing / len(df)) * 100
            logger.info(f"  Missing values: {missing:,} ({missing_pct:.1f}%)")
    
    logger.info(f"\n{'='*50}\n")

def generate_unique_id(row: pd.Series, prefix: str, id_fields: list) -> str:
    """
    Generate a unique ID following data science best practices.
    
    Args:
        row: DataFrame row containing the data
        prefix: Type prefix (e.g., 'MSG', 'CMT', 'POST')
        id_fields: List of fields to use for ID generation
    
    Returns:
        str: Unique identifier
    """
    # Create a string combining all relevant fields
    content_parts = []
    for field in id_fields:
        value = row.get(field, '')
        if pd.isna(value):
            value = ''
        elif isinstance(value, (pd.Timestamp, datetime)):
            # Format datetime to include microseconds for more precision
            value = value.strftime('%Y-%m-%d %H:%M:%S.%f')
        else:
            value = str(value)
        content_parts.append(value)
    
    content_str = '_'.join(content_parts)
    
    # Generate hash
    hash_object = hashlib.md5(content_str.encode())
    hash_hex = hash_object.hexdigest()[:12]  # Use first 12 characters of hash for more uniqueness
    
    # Combine prefix and hash
    return f"{prefix}_{hash_hex}"

class DataCleaner:
    def __init__(self):
        self.cleaning_stats = {}
        self.cleaned_files = set()
        self.id_mapping_file = 'id_mappings.json'
        self.id_mappings = self._load_id_mappings()

    def _load_id_mappings(self) -> dict:
        """Load ID mappings from JSON file."""
        if os.path.exists(self.id_mapping_file):
            try:
                with open(self.id_mapping_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error reading {self.id_mapping_file}, creating new mappings")
                return {
                    'messages': {},
                    'comments': {},
                    'posts': {},
                    'connections': {},
                    'likes': {}
                }
        return {
            'messages': {},
            'comments': {},
            'posts': {},
            'connections': {},
            'likes': {}
        }

    def _save_id_mappings(self) -> None:
        """Save ID mappings to JSON file."""
        with open(self.id_mapping_file, 'w') as f:
            json.dump(self.id_mappings, f, indent=2)

    def _update_id_mappings(self, df: pd.DataFrame, data_type: str) -> None:
        """Update ID mappings for a specific data type."""
        if data_type not in self.id_mappings:
            self.id_mappings[data_type] = {}

        # Get the appropriate ID column based on data type
        id_column = {
            'messages': 'message_id',
            'comments': 'comment_id',
            'posts': 'post_id',
            'connections': 'connection_id',
            'likes': 'like_id'
        }.get(data_type)

        if id_column and id_column in df.columns:
            # Create mapping of original data to unique IDs
            for _, row in df.iterrows():
                original_key = self._get_original_key(row, data_type)
                if original_key:
                    self.id_mappings[data_type][original_key] = row[id_column]

        self._save_id_mappings()

    def _get_original_key(self, row: pd.Series, data_type: str) -> str:
        """Get the original key for a row based on data type."""
        if data_type == 'messages':
            return f"{row['message_date']}_{row['content'][:100]}"  # Use first 100 chars of content
        elif data_type == 'comments':
            return f"{row['Date']}_{row['Link']}"
        elif data_type == 'posts':
            return row['ShareLink']
        elif data_type == 'connections':
            return row['URL']
        elif data_type == 'likes':
            return row['Link']
        return None

    def safe_to_datetime(self, df: pd.DataFrame, column: str) -> None:
        """Safely convert a column to datetime with UTC timezone, handling various formats and errors."""
        if column in df.columns:
            # Attempt to convert to datetime, coercing errors to NaT (Not a Time)
            df[column] = pd.to_datetime(df[column], errors='coerce')
            
            # Check if the column is already timezone-aware
            if hasattr(df[column].dtype, 'tz') and df[column].dtype.tz is not None:
                 # If tz-aware, convert to UTC
                df[column] = df[column].dt.tz_convert('UTC')
            else:
                 # If not tz-aware, localize to UTC, handling NaT values by filtering
                 # Filter out NaT values before tz_localize
                valid_dates = df[column].dropna()
                if not valid_dates.empty:
                    # Localize valid dates to UTC
                    localized_dates = valid_dates.dt.tz_localize('UTC')
                    # Update the original column with localized dates, leaving NaT as is
                    df[column] = localized_dates.reindex(df[column].index)

    def save_cleaning_status(self, table_name: str, file_path: str) -> None:
        """Save cleaning status to a JSON file."""
        status_file = 'cleaning_status.json'
        status = {
            'last_cleaned': datetime.now().isoformat(),
            'cleaned_files': list(self.cleaned_files),
            'stats': self.cleaning_stats
        }
        
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        logger.info(f"Cleaning status saved to {status_file}")

    def clean_messages(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """Clean messages dataset."""
        logger.info("Starting messages cleaning")
        try:
            df_clean = df.copy()
            initial_rows = len(df_clean)
            logger.info(f"Original dataset shape: {df_clean.shape}")
            
            # Rename columns to match database schema before cleaning
            column_mapping = {
                'CONVERSATION ID': 'conversation_id',
                'FROM': 'sender',
                'TO': 'recipient',
                'DATE': 'message_date',
                'SUBJECT': 'subject',
                'CONTENT': 'content'
            }
            df_clean = df_clean.rename(columns=column_mapping)
            
            # Convert message_date to datetime with timezone
            self.safe_to_datetime(df_clean, 'message_date')

            # Ensure text fields are strings and handle missing values
            text_columns = ['sender', 'recipient', 'subject', 'content']
            for col in text_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].astype(str).replace('nan', '').replace('None', '')
                    df_clean[col] = df_clean[col].str.strip()

            # Generate unique IDs for messages using only date and content
            df_clean['message_id'] = df_clean.apply(
                lambda row: generate_unique_id(row, 'MSG', ['message_date', 'content']), 
                axis=1
            )
            logger.info("Generated unique IDs for messages using date and content")

            # Update ID mappings
            self._update_id_mappings(df_clean, 'messages')

            # Remove duplicates using the new message_id
            df_clean = df_clean.drop_duplicates(subset=['message_id'])
            duplicates_removed = initial_rows - len(df_clean)
            logger.info(f"Removed {duplicates_removed} duplicate rows")

            # Save cleaned data
            cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
            df_clean.to_csv(cleaned_file_path, index=False)
            self.cleaned_files.add(cleaned_file_path)
            logger.info(f"Cleaned data saved to {cleaned_file_path}")

            return df_clean

        except Exception as e:
            logger.error(f"Error cleaning messages: {str(e)}")
            raise

    def clean_comments(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """Clean comments dataset."""
        logger.info("Starting comments cleaning")
        try:
            df_clean = df.copy()
            initial_rows = len(df_clean)
            logger.info(f"Original dataset shape: {df_clean.shape}")
            
            # Convert date column
            self.safe_to_datetime(df_clean, 'Date')
            
            # Drop rows with missing values in 'Date', 'Message', or 'Link'
            df_clean.dropna(subset=["Date", "Message", "Link"], inplace=True)
            rows_after_dropping_na = len(df_clean)
            logger.info(f"Dropped {initial_rows - rows_after_dropping_na} rows with missing 'Date', 'Message', or 'Link'.")
            
            # Extract year, month, and year-month for trend analysis
            df_clean["Year"] = df_clean["Date"].dt.year
            df_clean["Month"] = df_clean["Date"].dt.month
            df_clean["Year-Month"] = df_clean["Date"].dt.to_period("M")

            # Clean text fields
            if 'Message' in df_clean.columns:
                df_clean['Message'] = df_clean['Message'].fillna('')
            
            # Generate unique IDs for comments using Date and Link only
            df_clean['comment_id'] = df_clean.apply(
                lambda row: generate_unique_id(row, 'CMT', ['Date', 'Link']), 
                axis=1
            )
            logger.info("Generated unique IDs for comments using Date and Link")
            
            # Remove duplicates using the new comment_id
            df_clean = df_clean.drop_duplicates(subset=['comment_id'])
            duplicates_removed = initial_rows - len(df_clean)
            logger.info(f"Removed {duplicates_removed} duplicate rows")
            
            # Log cleaning statistics
            self.cleaning_stats['comments'] = {
                'original_rows': len(df),
                'cleaned_rows': len(df_clean),
                'removed_rows': len(df) - len(df_clean)
            }
            
            # Save cleaned data
            cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
            df_clean.to_csv(cleaned_file_path, index=False)
            self.cleaned_files.add(cleaned_file_path)
            logger.info(f"Cleaned data saved to {cleaned_file_path}")

            # Save cleaning status
            self.save_cleaning_status('comments', cleaned_file_path)
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning comments: {str(e)}")
            raise

    def clean_posts(self, df, file_path):
        """Clean posts data from Shares.csv"""
        logger.info("\nStarting posts cleaning from Shares data...")
        
        # Create a copy to avoid SettingWithCopyWarning
        df_clean = df.copy()
        
        # Drop columns with many missing values
        columns_to_drop = ["SharedUrl", "MediaUrl"]
        df_clean = df_clean.drop(columns=columns_to_drop)
        
        # Convert date to datetime
        df_clean['date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        
        # Drop rows with missing values in required fields
        required_fields = ['Date', 'ShareLink', 'ShareCommentary', 'Visibility']
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=required_fields)
        rows_dropped = initial_rows - len(df_clean)
        
        # Extract year, month, and year-month
        df_clean['year'] = df_clean['date'].dt.year
        df_clean['month'] = df_clean['date'].dt.month
        df_clean['year_month'] = df_clean['date'].dt.strftime('%Y-%m')
        
        # Clean text fields and rename columns
        text_fields = {
            'post_id': 'ShareLink',  # Use ShareLink as post_id
            'post_commentary': 'ShareCommentary',
            'post_link': 'ShareLink',
            'visibility': 'Visibility'
        }
        for new_col, old_col in text_fields.items():
            df_clean.loc[:, new_col] = df_clean[old_col].fillna('')
            logger.info(f"Cleaned text field: {old_col} -> {new_col}")
        
        # Remove duplicates using ShareLink as the unique identifier
        df_clean = df_clean.drop_duplicates(subset=['post_id'])
        duplicates_removed = initial_rows - len(df_clean)
        logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Keep only required columns
        final_columns = ['post_id', 'date', 'post_link', 'post_commentary', 'visibility', 'year', 'month', 'year_month']
        df_clean = df_clean[final_columns]
        logger.info(f"Final columns selected: {final_columns}")
        
        # Save cleaned data
        output_file = file_path.replace('Shares.csv', 'Posts_cleaned.csv')
        df_clean.to_csv(output_file, index=False)
        logger.info(f"Cleaned posts data saved to {output_file}")
        
        # Record cleaning stats
        self.cleaning_stats['posts'] = {
            'original_rows': df.shape[0],
            'rows_after_cleaning': df_clean.shape[0],
            'missing_values_dropped': rows_dropped,
            'duplicates_removed': duplicates_removed,
            'columns_removed': len(columns_to_drop),
            'final_columns': final_columns
        }
        
        # Save cleaning status
        self.save_cleaning_status('posts', output_file)
        
        logger.info(f"Posts cleaning completed. Final shape: {df_clean.shape}")
        return df_clean

    def clean_likes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean likes dataset."""
        logger.info("Starting likes cleaning")
        try:
            df_clean = df.copy()
            logger.info(f"Original dataset shape: {df_clean.shape}")
            initial_rows = len(df_clean)
            
            # Convert date column to datetime
            self.safe_to_datetime(df_clean, 'Date')
            
            # Filter only 'LIKE' type reactions
            df_clean = df_clean[df_clean["Type"] == "LIKE"]
            logger.info(f"Filtered to only LIKE reactions. Remaining rows: {len(df_clean)}")
            
            # Add year and month columns for analysis
            df_clean["Year"] = df_clean["Date"].dt.year
            df_clean["Month"] = df_clean["Date"].dt.month
            df_clean["Year-Month"] = df_clean["Date"].dt.to_period("M")
            
            # Clean the Link column and use it as like_id
            if 'Link' in df_clean.columns:
                df_clean['Link'] = df_clean['Link'].fillna('')
                df_clean['Link'] = df_clean['Link'].astype(str).str.strip()
                df_clean['like_id'] = df_clean['Link']  # Use Link as like_id
                logger.info("Using Link as like_id")
            
            # Remove duplicates using Link as the unique identifier
            df_clean = df_clean.drop_duplicates(subset=['like_id'])
            
            self.cleaning_stats['likes'] = {
                'original_rows': initial_rows,
                'cleaned_rows': len(df_clean),
                'removed_rows': initial_rows - len(df_clean),
                'like_type_count': len(df_clean[df_clean["Type"] == "LIKE"]),
                'years_covered': sorted(df_clean["Year"].unique().tolist())
            }
            
            logger.info(f"Likes cleaning completed. Removed {initial_rows - len(df_clean)} rows")
            logger.info(f"Years covered in the dataset: {sorted(df_clean['Year'].unique().tolist())}")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning likes: {str(e)}")
            raise

    def clean_connections(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """Clean connections dataset."""
        logger.info("Starting connections cleaning")
        try:
            df_clean = df.copy()
            logger.info(f"Original dataset shape: {df_clean.shape}")
            initial_rows = len(df_clean)
            
            # Rename columns to match database schema before cleaning
            column_mapping = {
                'First Name': 'first_name',
                'Last Name': 'last_name',
                'URL': 'connection_id',  # Use URL as connection_id
                'Email Address': 'email_address',
                'Company': 'company',
                'Position': 'position',
                'Connected On': 'connected_on'
            }
            df_clean = df_clean.rename(columns=column_mapping, inplace=False)

            # Convert connected_on to datetime with timezone
            self.safe_to_datetime(df_clean, 'connected_on')
            
            # Drop rows where only connected_on is present and all other fields are null
            mask = df_clean['connected_on'].notna() & df_clean.drop('connected_on', axis=1).isna().all(axis=1)
            rows_to_drop = mask.sum()
            df_clean = df_clean[~mask]
            
            # Ensure text fields are strings and handle missing values
            text_columns = ['first_name', 'last_name', 'connection_id', 'email_address', 'company', 'position']
            for col in text_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].astype(str).replace('nan', '').replace('None', '')
                    df_clean[col] = df_clean[col].str.strip()
            
            # Remove duplicates using connection_id
            df_clean = df_clean.drop_duplicates(subset=['connection_id'])
            duplicates_removed = initial_rows - len(df_clean)
            
            self.cleaning_stats['connections'] = {
                'original_rows': len(df),
                'cleaned_rows': len(df_clean),
                'removed_rows': len(df) - len(df_clean)
            }
            
            # Save cleaned data
            cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
            df_clean.to_csv(cleaned_file_path, index=False)
            self.cleaned_files.add(cleaned_file_path)
            
            # Save cleaning status
            self.save_cleaning_status('connections', cleaned_file_path)
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning connections: {str(e)}")
            raise

    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Return cleaning statistics."""
        return self.cleaning_stats

def main():
    """Main function to demonstrate usage."""
    try:
        logger.info("\n" + "="*50)
        logger.info("Starting Data Cleaning Process")
        logger.info("="*50 + "\n")
        
        cleaner = DataCleaner()
        
        # Process messages
        messages_file = 'data/messages.csv'
        if os.path.exists(messages_file):
            logger.info(f"\nProcessing messages file: {messages_file}")
            df_messages = pd.read_csv(messages_file)
            # Log initial statistics
            log_dataset_stats(df_messages, "Before Cleaning", "Messages")
            df_cleaned_messages = cleaner.clean_messages(df_messages, messages_file)
            # Log final statistics
            log_dataset_stats(df_cleaned_messages, "After Cleaning", "Messages")
        else:
            logger.warning(f"Messages file not found: {messages_file}")
        
        # Process connections
        connections_file = 'data/Connections.csv'
        if os.path.exists(connections_file):
            logger.info(f"\nProcessing connections file: {connections_file}")
            df_connections = pd.read_csv(connections_file, skiprows=3)
            # Log initial statistics
            log_dataset_stats(df_connections, "Before Cleaning", "Connections")
            df_cleaned_connections = cleaner.clean_connections(df_connections, connections_file)
            # Log final statistics
            log_dataset_stats(df_cleaned_connections, "After Cleaning", "Connections")
        else:
            logger.warning(f"Connections file not found: {connections_file}")
        
        # Process comments
        comments_file = 'data/Comments.csv'
        if os.path.exists(comments_file):
            logger.info(f"\nProcessing comments file: {comments_file}")
            rows = []
            with open(comments_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if len(row) > 3:
                        date, link = row[0], row[1]
                        message = ','.join(row[2:])
                        rows.append([date, link, message])
                    else:
                        rows.append(row)
            
            df_comments = pd.DataFrame(rows, columns=header)
            # Log initial statistics
            log_dataset_stats(df_comments, "Before Cleaning", "Comments")
            df_cleaned_comments = cleaner.clean_comments(df_comments, comments_file)
            # Log final statistics
            log_dataset_stats(df_cleaned_comments, "After Cleaning", "Comments")
        else:
            logger.warning(f"Comments file not found: {comments_file}")
        
        # Process shares/posts
        shares_file = 'data/Shares.csv'
        if os.path.exists(shares_file):
            logger.info(f"\nProcessing shares file: {shares_file}")
            df_shares = pd.read_csv(shares_file)
            # Log initial statistics
            log_dataset_stats(df_shares, "Before Cleaning", "Posts")
            df_cleaned_posts = cleaner.clean_posts(df_shares, shares_file)
            # Log final statistics
            log_dataset_stats(df_cleaned_posts, "After Cleaning", "Posts")
        else:
            logger.warning(f"Shares file not found: {shares_file}")
        
        # Process reactions
        reactions_file = 'data/Reactions.csv'
        if os.path.exists(reactions_file):
            logger.info(f"\nProcessing reactions file: {reactions_file}")
            df_reactions = pd.read_csv(reactions_file)
            # Log initial statistics
            log_dataset_stats(df_reactions, "Before Cleaning", "Likes")
            df_cleaned_reactions = cleaner.clean_likes(df_reactions)
            # Log final statistics
            log_dataset_stats(df_cleaned_reactions, "After Cleaning", "Likes")
            cleaned_reactions_file = reactions_file.replace('.csv', '_cleaned.csv')
            df_cleaned_reactions.to_csv(cleaned_reactions_file, index=False)
            cleaner.cleaned_files.add(cleaned_reactions_file)
            logger.info(f"Cleaned reactions saved to {cleaned_reactions_file}")
        else:
            logger.warning(f"Reactions file not found: {reactions_file}")
        
        logger.info("\n" + "="*50)
        logger.info("Data Cleaning Process Completed Successfully")
        logger.info("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        raise

if __name__ == "__main__":
    main() 