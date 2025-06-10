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

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_cleaning.log')
    ]
)
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        self.cleaning_stats = {}
        self.cleaned_files = set()

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
            logger.info(f"Original dataset shape: {df_clean.shape}")
            initial_rows = len(df_clean)

            # Rename columns to match database schema before cleaning
            column_mapping = {
                'Conversation Id': 'conversation_id',
                'From': 'sender',
                'To': 'recipient',
                'Date': 'message_date',
                'Subject': 'subject',
                'Content': 'content'
            }
            # Use inplace=False as we are assigning the result back to df_clean
            df_clean = df_clean.rename(columns=column_mapping, inplace=False)
            logger.info("Columns renamed to match database schema")
            logger.info(f"Current columns: {df_clean.columns.tolist()}")

            # Convert message_date to datetime with timezone
            # The safe_to_datetime method modifies the DataFrame in place
            self.safe_to_datetime(df_clean, 'message_date')

            # Ensure text fields are strings and handle missing values
            text_columns = ['sender', 'recipient', 'subject', 'content']
            for col in text_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].astype(str).replace('nan', '').replace('None', '')
                    df_clean[col] = df_clean[col].str.strip()
            logger.info("Text fields cleaned and whitespace removed")

            # Remove duplicates
            df_clean = df_clean.drop_duplicates()
            duplicates_removed = initial_rows - len(df_clean)
            logger.info(f"Removed {duplicates_removed} duplicate rows")

            # Log cleaning statistics
            # Ensure column names in stats reflect the database schema names
            self.cleaning_stats['messages'] = {
                'original_rows': len(df),
                'cleaned_rows': len(df_clean),
                'removed_rows': len(df) - len(df_clean)
                 # We are no longer removing columns in this step, so columns_removed might be misleading.
                 # Removing this for clarity or updating the logic if columns are dropped.
            }
            
            # Save cleaned data
            # Ensure the saved CSV uses the database column names
            cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
            df_clean.to_csv(cleaned_file_path, index=False)
            self.cleaned_files.add(cleaned_file_path)
            logger.info(f"Cleaned data saved to {cleaned_file_path}")

            # Save cleaning status
            self.save_cleaning_status('messages', cleaned_file_path)

            logger.info(f"Messages cleaning completed. Final shape: {df_clean.shape}")
            return df_clean # Explicitly return the cleaned DataFrame

        except Exception as e:
            logger.error(f"Error cleaning messages: {str(e)}")
            raise

    def clean_comments(self, df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        """Clean comments dataset."""
        logger.info("Starting comments cleaning")
        try:
            df_clean = df.copy()
            logger.info(f"Original dataset shape: {df_clean.shape}")
            initial_rows = len(df_clean)
            
            # Convert date column
            self.safe_to_datetime(df_clean, 'Date')
            logger.info("Date column converted to datetime")
            
            # Drop rows with missing values in 'Date', 'Message', or 'Link'
            df_clean.dropna(subset=["Date", "Message", "Link"], inplace=True)
            rows_after_dropping_na = len(df_clean)
            logger.info(f"Dropped {initial_rows - rows_after_dropping_na} rows with missing 'Date', 'Message', or 'Link'.")
            
            # Extract year, month, and year-month for trend analysis
            df_clean["Year"] = df_clean["Date"].dt.year
            df_clean["Month"] = df_clean["Date"].dt.month
            df_clean["Year-Month"] = df_clean["Date"].dt.to_period("M")
            logger.info("Extracted year, month, and year-month from date")

            # Clean text fields
            if 'Message' in df_clean.columns:
                df_clean['Message'] = df_clean['Message'].fillna('')
                logger.info("Cleaned Message field")
            
            # Remove duplicates
            df_clean = df_clean.drop_duplicates()
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
            
            logger.info(f"Comments cleaning completed. Final shape: {df_clean.shape}")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning comments: {str(e)}")
            raise

    def clean_posts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean posts dataset."""
        logger.info("Starting posts cleaning")
        try:
            df_clean = df.copy()
            
            # Convert date column
            self.safe_to_datetime(df_clean, 'Date')
            
            # Clean text fields
            text_columns = ['Content', 'Title']
            for col in text_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna('')
            
            # Remove duplicates
            df_clean = df_clean.drop_duplicates()
            
            self.cleaning_stats['posts'] = {
                'original_rows': len(df),
                'cleaned_rows': len(df_clean),
                'removed_rows': len(df) - len(df_clean)
            }
            
            logger.info(f"Posts cleaning completed. Removed {len(df) - len(df_clean)} rows")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning posts: {str(e)}")
            raise

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
            
            # Remove duplicates
            df_clean = df_clean.drop_duplicates()
            
            # Clean the Link column
            if 'Link' in df_clean.columns:
                df_clean['Link'] = df_clean['Link'].fillna('')
                df_clean['Link'] = df_clean['Link'].astype(str).str.strip()
            
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
                'URL': 'url',
                'Email Address': 'email_address',
                'Company': 'company',
                'Position': 'position',
                'Connected On': 'connected_on'
            }
            # Use inplace=False as we are assigning the result back to df_clean
            df_clean = df_clean.rename(columns=column_mapping, inplace=False)
            logger.info("Columns renamed to match database schema")
            logger.info(f"Current columns: {df_clean.columns.tolist()}")

            # Convert connected_on to datetime with timezone
            # The safe_to_datetime method modifies the DataFrame in place
            self.safe_to_datetime(df_clean, 'connected_on')
            
            # Drop rows where only connected_on is present and all other fields are null
            # Use database column name 'connected_on'
            mask = df_clean['connected_on'].notna() & df_clean.drop('connected_on', axis=1).isna().all(axis=1)
            rows_to_drop = mask.sum()
            df_clean = df_clean[~mask]
            logger.info(f"Removed {rows_to_drop} rows where only connected_on was present")
            
            # Ensure text fields are strings and handle missing values
            text_columns = ['first_name', 'last_name', 'url', 'email_address', 'company', 'position']
            for col in text_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].astype(str).replace('nan', '').replace('None', '')
                    df_clean[col] = df_clean[col].str.strip()
            logger.info("Text fields cleaned and whitespace removed")
            
            # Remove duplicates
            df_clean = df_clean.drop_duplicates()
            duplicates_removed = initial_rows - len(df_clean)
            logger.info(f"Removed {duplicates_removed} duplicate rows")
            
            self.cleaning_stats['connections'] = {
                'original_rows': len(df),
                'cleaned_rows': len(df_clean),
                'removed_rows': len(df) - len(df_clean)
            }
            
            # Save cleaned data
            # Ensure the saved CSV uses the database column names
            cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
            df_clean.to_csv(cleaned_file_path, index=False)
            self.cleaned_files.add(cleaned_file_path)
            logger.info(f"Cleaned data saved to {cleaned_file_path}")
            
            # Save cleaning status
            self.save_cleaning_status('connections', cleaned_file_path)
            
            logger.info(f"Connections cleaning completed. Final shape: {df_clean.shape}")
            return df_clean # Explicitly return the cleaned DataFrame
            
        except Exception as e:
            logger.error(f"Error cleaning connections: {str(e)}")
            raise

    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Return cleaning statistics."""
        return self.cleaning_stats

def main():
    """Main function to demonstrate usage."""
    try:
        cleaner = DataCleaner()
        
        # Process messages
        messages_file = 'data/messages.csv'
        if os.path.exists(messages_file):
            logger.info(f"Processing messages file: {messages_file}")
            df_messages = pd.read_csv(messages_file)
            cleaner.clean_messages(df_messages, messages_file)
        else:
            logger.warning(f"Messages file not found: {messages_file}")
        
        # Process connections
        connections_file = 'data/Connections.csv'
        if os.path.exists(connections_file):
            logger.info(f"Processing connections file: {connections_file}")
            # Read Connections.csv with skiprows=3 to skip the header information
            df_connections = pd.read_csv(connections_file, skiprows=3)
            cleaner.clean_connections(df_connections, connections_file)
        else:
            logger.warning(f"Connections file not found: {connections_file}")
        
        # Process comments
        comments_file = 'data/Comments.csv'
        if os.path.exists(comments_file):
            logger.info(f"Processing comments file: {comments_file}")
            # Read Comments.csv using csv module for more robust parsing
            rows = []
            with open(comments_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Get header row
                for row in reader:
                    if len(row) > 3:  # If we have more than 3 fields, combine the extra ones into the message
                        date, link = row[0], row[1]
                        message = ','.join(row[2:])  # Combine all remaining fields into the message
                        rows.append([date, link, message])
                    else:
                        rows.append(row)
            
            # Convert to DataFrame
            df_comments = pd.DataFrame(rows, columns=header)
            cleaner.clean_comments(df_comments, comments_file)
        else:
            logger.warning(f"Comments file not found: {comments_file}")
        
        # Process reactions
        reactions_file = 'data/Reactions.csv'
        if os.path.exists(reactions_file):
            logger.info(f"Processing reactions file: {reactions_file}")
            df_reactions = pd.read_csv(reactions_file)
            df_cleaned_reactions = cleaner.clean_likes(df_reactions)
            # Save cleaned reactions
            cleaned_reactions_file = reactions_file.replace('.csv', '_cleaned.csv')
            df_cleaned_reactions.to_csv(cleaned_reactions_file, index=False)
            cleaner.cleaned_files.add(cleaned_reactions_file)
            logger.info(f"Cleaned reactions saved to {cleaned_reactions_file}")
        else:
            logger.warning(f"Reactions file not found: {reactions_file}")
        
        logger.info("Data cleaning completed successfully")
        
    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        raise

if __name__ == "__main__":
    main() 