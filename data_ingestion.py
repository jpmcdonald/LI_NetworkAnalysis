import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Date, Text, TIMESTAMP, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime, UTC
import logging
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file
load_dotenv()

# Setup logging: log only to file with UTF-8 encoding (no console handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('linkedin_ingestion.log', encoding='utf-8')
    ]
)

def get_db_credentials():
    """Get database credentials from environment variables."""
    required_vars = ['POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    return {
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD'),
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5433'),
        'dbname': os.getenv('POSTGRES_DB')
    }

# Get database credentials
try:
    db_creds = get_db_credentials()
    DATABASE_URL = f"postgresql://{db_creds['user']}:{db_creds['password']}@{db_creds['host']}:{db_creds['port']}/{db_creds['dbname']}"
except ValueError as e:
    logging.error(f"Database configuration error: {e}")
    raise

Base = declarative_base()

class Connection(Base):
    __tablename__ = 'connections'
    id = Column(Integer, primary_key=True)
    first_name = Column(String(100))
    last_name = Column(String(100))
    url = Column(String(300), unique=True)
    email_address = Column(String(200))
    company = Column(String(200))
    position = Column(String(200))
    connected_on = Column(Date)
    cdc_datetime = Column(TIMESTAMP, nullable=False)
    notes = Column(Text)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
    deleted_at = Column(TIMESTAMP)
    __table_args__ = (UniqueConstraint('url', name='uix_url'),)

class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(100))
    sender = Column(String(200))
    recipient = Column(Text)
    message_date = Column(TIMESTAMP)
    subject = Column(Text)
    content = Column(Text)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
    deleted_at = Column(TIMESTAMP)

# Create engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def parse_date(date_str):
    """Parse date string to datetime object."""
    try:
        return datetime.strptime(date_str, '%d %b %Y').date()
    except Exception:
        return None

def format_recipients(recipients_str):
    """Format recipients string to be more database-friendly."""
    if not recipients_str:
        return ""
    try:
        recipients = [r.strip() for r in recipients_str.split(',')]
        if len(recipients) > 3:
            return f"{recipients[0]}, {recipients[1]}, {recipients[2]} and {len(recipients) - 3} others"
        return ", ".join(recipients)
    except Exception as e:
        logging.error(f"Error formatting recipients: {e}")
        return recipients_str

def ingest_data(df, model_class, batch_size=1000, skip_rows=0, is_update=False):
    """Generic function to ingest data into any table."""
    batch_time = datetime.now(UTC)
    total_inserted = 0
    total_skipped = 0
    total_rows = len(df)
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch = df.iloc[batch_start:batch_end]
        logging.info(f'Processing batch {batch_start//batch_size + 1} of {(total_rows + batch_size - 1)//batch_size} (rows {batch_start+1}-{batch_end})')
        session = Session()
        inserted, skipped = 0, 0
        new_records = []
        try:
            for _, row in batch.iterrows():
                data = row.to_dict()
                if 'created_at' not in data:
                    data['created_at'] = batch_time
                if 'updated_at' not in data:
                    data['updated_at'] = batch_time
                if model_class == Connection and 'cdc_datetime' not in data:
                    data['cdc_datetime'] = batch_time
                try:
                    if model_class == Message:
                        if not data['sender'] or not data['recipient']:
                            logging.warning(f"Skipping message with missing sender/recipient: {data}")
                            skipped += 1
                            continue
                        conversation_id = str(data['conversation_id']).strip()
                        sender = str(data['sender']).strip()
                        recipient = format_recipients(str(data['recipient']).strip())
                        subject = str(data['subject']).strip()
                        content = str(data['content']).strip()
                        try:
                            message = Message(
                                conversation_id=conversation_id,
                                sender=sender,
                                recipient=recipient,
                                message_date=data['message_date'],
                                subject=subject,
                                content=content,
                                created_at=data['created_at'],
                                updated_at=data['updated_at']
                            )
                            session.add(message)
                            session.flush()
                            inserted += 1
                            new_records.append(f"Message from {sender} to {len(recipient.split(','))} recipients")
                        except Exception as e:
                            logging.error(f'Error creating message object: {e}')
                            logging.error(f'Data that caused error: {data}')
                            session.rollback()
                            skipped += 1
                            continue
                    else:
                        stmt = insert(model_class).values(**data)
                        if model_class == Connection:
                            stmt = stmt.on_conflict_do_nothing(index_elements=['url'])
                        else:
                            stmt = stmt.on_conflict_do_nothing(index_elements=['id'])
                        result = session.execute(stmt)
                        if result.rowcount:
                            inserted += 1
                            if model_class == Connection:
                                new_records.append(f"{data['first_name']} {data['last_name']} ({data['company']})")
                        else:
                            skipped += 1
                except Exception as e:
                    logging.error(f'Error inserting row: {e} | Data: {data}')
                    skipped += 1
            session.commit()
            total_inserted += inserted
            total_skipped += skipped
            logging.info(f'Batch {batch_start//batch_size + 1} complete at {batch_time}')
            logging.info(f'Batch rows processed: {len(batch)}')
            logging.info(f'Batch new records: {inserted}')
            logging.info(f'Batch skipped: {skipped}')
            if new_records:
                logging.info('New records in this batch:')
                for record in new_records:
                    logging.info(f'  - {record}')
        except Exception as e:
            logging.error(f'Error processing batch: {e}')
            session.rollback()
        finally:
            session.close()
    return total_inserted, total_skipped, total_rows

def process_dataset(file_path, model_class, column_mapping, date_columns=None, skip_rows=0, batch_size=1000):
    """Process a single dataset with its specific configuration."""
    batch_time = datetime.now(UTC)
    logging.info(f'Starting ingestion for {model_class.__tablename__} at {batch_time}')
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path, skiprows=skip_rows)
        logging.info(f'Successfully read CSV file with {len(df)} rows')
        
        # For messages, we need to handle the data differently
        if model_class == Message:
            # Log the first few rows of raw data for debugging
            logging.info("First few rows of raw message data:")
            logging.info(df[['CONVERSATION ID', 'FROM', 'TO', 'DATE', 'SUBJECT', 'CONTENT']].head())
            
            # Clean and transform the data before renaming columns
            df['message_date'] = pd.to_datetime(df['DATE'], errors='coerce')
            df['sender'] = df['FROM'].fillna('')
            df['recipient'] = df['TO'].fillna('')
            df['subject'] = df['SUBJECT'].fillna('')
            df['content'] = df['CONTENT'].fillna('')
            df['conversation_id'] = df['CONVERSATION ID'].fillna('')
            
            # Validate data before transformation
            logging.info("Data validation before transformation:")
            logging.info(f"Null values in FROM: {df['FROM'].isnull().sum()}")
            logging.info(f"Null values in TO: {df['TO'].isnull().sum()}")
            logging.info(f"Null values in CONTENT: {df['CONTENT'].isnull().sum()}")
            
            # Log the transformed data for debugging
            logging.info("First few rows of transformed message data:")
            logging.info(df[['conversation_id', 'sender', 'recipient', 'message_date', 'subject', 'content']].head())
            
            # Drop any extra columns we don't need
            columns_to_keep = ['conversation_id', 'sender', 'recipient', 'message_date', 'subject', 'content']
            df = df[columns_to_keep]
            
            # Verify no null values in required fields
            null_counts = df.isnull().sum()
            logging.info("Null value counts in transformed data:")
            logging.info(null_counts)
            
            # Validate data types
            logging.info("Data types after transformation:")
            logging.info(df.dtypes)
            
            logging.info('Messages data cleaned and transformed')
        else:
            # For connections, use the original column mapping
            df = df.rename(columns=column_mapping)
            logging.info('Columns renamed to match database schema')
        
        # Parse dates
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        logging.info('Dates parsed')
        
        # Process in batches
        total_inserted, total_skipped, total_rows = ingest_data(df, model_class, batch_size, skip_rows)
        
        # Log final summary
        end_time = datetime.now(UTC)
        duration = end_time - batch_time
        logging.info(f'=== {model_class.__tablename__.title()} Ingestion Complete ===')
        logging.info(f'Total time: {duration}')
        logging.info(f'Total rows processed: {total_rows}')
        logging.info(f'Total new records: {total_inserted}')
        logging.info(f'Total skipped: {total_skipped}')
        logging.info(f'Success rate: {(total_inserted + total_skipped) / total_rows * 100:.2f}%')
        
        return True
        
    except Exception as e:
        logging.error(f'Error processing {model_class.__tablename__}: {e}')
        return False

def main():
    parser = argparse.ArgumentParser(description='Ingest LinkedIn connections and messages into database')
    parser.add_argument('--data-dir', default='data',
                      help='Directory containing cleaned CSV files (default: data)')
    parser.add_argument('--batch-size', type=int, default=1000,
                      help='Number of records to process in each batch (default: 1000)')
    args = parser.parse_args()
    datasets = {
        'connections': {
            'file': 'Connections_cleaned.csv',
            'model': Connection,
            'column_mapping': {
                'First Name': 'first_name',
                'Last Name': 'last_name',
                'URL': 'url',
                'Email Address': 'email_address',
                'Company': 'company',
                'Position': 'position',
                'Connected On': 'connected_on'
            },
            'date_columns': ['connected_on'],
            'skip_rows': 0
        },
        'messages': {
            'file': 'messages_cleaned.csv',
            'model': Message,
            'column_mapping': {},
            'date_columns': ['message_date'],
            'skip_rows': 0
        }
    }
    for dataset_name, config in datasets.items():
        file_path = os.path.join(args.data_dir, config['file'])
        if os.path.exists(file_path):
            logging.info(f'Processing {dataset_name} from {file_path}')
            success = process_dataset(
                file_path,
                config['model'],
                config['column_mapping'],
                config['date_columns'],
                config['skip_rows'],
                args.batch_size
            )
            if not success:
                logging.error(f'Failed to process {dataset_name}')
        else:
            logging.warning(f'File not found: {file_path}')

if __name__ == '__main__':
    main()
