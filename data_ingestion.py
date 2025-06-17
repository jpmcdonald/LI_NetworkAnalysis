import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Date, Text, TIMESTAMP, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime, UTC
import logging
from dotenv import load_dotenv
import argparse
from data_cleaning import DataCleaner
import time
from datetime import timezone

# Load environment variables from .env file
load_dotenv()

# Setup logging: log to both console and file with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),  # Add console handler
        logging.FileHandler('data_ingestion.log', encoding='utf-8')
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
    message_id = Column(String(100), unique=True)
    sender = Column(String(200))
    recipient = Column(Text)
    message_date = Column(TIMESTAMP)
    subject = Column(Text)
    content = Column(Text)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
    deleted_at = Column(TIMESTAMP)
    __table_args__ = (UniqueConstraint('message_id', name='uix_message_id'),)

class Like(Base):
    __tablename__ = 'likes'
    id = Column(Integer, primary_key=True)
    date = Column(TIMESTAMP)
    type = Column(String(50))
    link = Column(String(300), unique=True)
    year = Column(Integer)
    month = Column(Integer)
    year_month = Column(String(7))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
    deleted_at = Column(TIMESTAMP)
    __table_args__ = (UniqueConstraint('link', name='uix_like_link'),)

class Comment(Base):
    __tablename__ = 'comments'
    id = Column(Integer, primary_key=True)
    comment_id = Column(String(100), unique=True)
    date = Column(TIMESTAMP)
    message = Column(Text)
    year = Column(Integer)
    month = Column(Integer)
    year_month = Column(String(7))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
    deleted_at = Column(TIMESTAMP)
    __table_args__ = (UniqueConstraint('comment_id', name='uix_comment_id'),)

class Post(Base):
    __tablename__ = 'posts'
    id = Column(Integer, primary_key=True)
    date = Column(TIMESTAMP)
    post_link = Column(String(300), unique=True)
    post_commentary = Column(Text)
    visibility = Column(String(50))
    year = Column(Integer)
    month = Column(Integer)
    year_month = Column(String(7))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
    deleted_at = Column(TIMESTAMP)
    __table_args__ = (UniqueConstraint('post_link', name='uix_post_link'),)

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

    # Get the appropriate unique identifier field based on model type
    unique_id_field = {
        Connection: 'url',
        Message: 'message_id',
        Comment: 'comment_id',
        Post: 'post_link',
        Like: 'link'
    }.get(model_class)

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
                        message_id = str(data['message_id']).strip()
                        sender = str(data['sender']).strip()
                        recipient = format_recipients(str(data['recipient']).strip())
                        subject = str(data['subject']).strip()
                        content = str(data['content']).strip()
                        try:
                            message = Message(
                                message_id=message_id,
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
                    elif model_class == Like:
                        try:
                            # Create a new Like object with the transformed data
                            like = Like(
                                date=data['date'],
                                type=str(data['type']),
                                link=str(data['link']),
                                year=int(data['year']),
                                month=int(data['month']),
                                year_month=str(data['year_month']),
                                created_at=data['created_at'],
                                updated_at=data['updated_at']
                            )
                            session.add(like)
                            session.flush()
                            inserted += 1
                            new_records.append(f"Like on {data['date']} for {data['link']}")
                        except Exception as e:
                            logging.error(f'Error creating like object: {e}')
                            logging.error(f'Data that caused error: {data}')
                            session.rollback()
                            skipped += 1
                            continue
                    elif model_class == Comment:
                        try:
                            comment = Comment(
                                comment_id=str(data['comment_id']),
                                date=data['date'],
                                message=str(data['message']),
                                year=int(data['year']),
                                month=int(data['month']),
                                year_month=str(data['year_month']),
                                created_at=data['created_at'],
                                updated_at=data['updated_at']
                            )
                            session.add(comment)
                            session.flush()
                            inserted += 1
                            new_records.append(f"Comment on {data['date']}")
                        except Exception as e:
                            logging.error(f'Error creating comment object: {e}')
                            logging.error(f'Data that caused error: {data}')
                            session.rollback()
                            skipped += 1
                            continue
                    elif model_class == Post:
                        try:
                            post = Post(
                                date=data['date'],
                                post_link=str(data['post_link']),
                                post_commentary=str(data['post_commentary']),
                                visibility=str(data['visibility']),
                                year=int(data['year']),
                                month=int(data['month']),
                                year_month=str(data['year_month']),
                                created_at=data['created_at'],
                                updated_at=data['updated_at']
                            )
                            session.add(post)
                            session.flush()
                            inserted += 1
                            new_records.append(f"Post on {data['date']} for {data['post_link']}")
                        except Exception as e:
                            logging.error(f'Error creating post object: {e}')
                            logging.error(f'Data that caused error: {data}')
                            session.rollback()
                            skipped += 1
                            continue
                    else:
                        # For other models, use the insert statement with unique constraint handling
                        stmt = insert(model_class).values(**data)
                        if unique_id_field:
                            stmt = stmt.on_conflict_do_nothing(index_elements=[unique_id_field])
                        else:
                            stmt = stmt.on_conflict_do_nothing(index_elements=['id'])
                        result = session.execute(stmt)
                        if result.rowcount > 0:
                            inserted += 1
                            new_records.append(f"{model_class.__name__} record")
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
    batch_time = time.time()
    total_records = 0
    total_errors = 0
    total_skipped = 0
    
    # Create a new session for this dataset
    session = Session()
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, skiprows=skip_rows)
        logging.info(f"Processing {len(df)} records from {file_path}")
        
        # Apply column mapping if provided
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Convert date columns
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        
        # Process in batches
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            batch_records = 0
            batch_errors = 0
            batch_skipped = 0
            
            for _, row in batch.iterrows():
                try:
                    # Convert row to dict and handle NaN values
                    data = {k: (None if pd.isna(v) else v) for k, v in row.items()}
                    
                    # Handle connection_id field for Connection model
                    if model_class == Connection and 'connection_id' in data:
                        data['url'] = data.pop('connection_id')
                    
                    # Remove conversation_id from Message data
                    if model_class == Message and 'conversation_id' in data:
                        data.pop('conversation_id')
                    
                    # For Message model, only keep the fields we need
                    if model_class == Message:
                        required_fields = ['message_id', 'sender', 'recipient', 'message_date', 'subject', 'content']
                        data = {k: v for k, v in data.items() if k in required_fields}
                    # For Like model, only keep the fields we need
                    elif model_class == Like:
                        required_fields = ['date', 'type', 'link']
                        data = {k: v for k, v in data.items() if k in required_fields}
                    # For Comment model, only keep the fields we need
                    elif model_class == Comment:
                        required_fields = ['date', 'message', 'comment_id']
                        data = {k: v for k, v in data.items() if k in required_fields}
                    # For Post model, only keep the fields we need
                    elif model_class == Post:
                        required_fields = ['date', 'post_link', 'post_commentary', 'visibility']
                        data = {k: v for k, v in data.items() if k in required_fields}
                    
                    # Add temporal columns if they don't exist
                    if 'date' in data and data['date'] is not None:
                        date = pd.to_datetime(data['date'])
                        data['year'] = date.year
                        data['month'] = date.month
                        data['year_month'] = f"{date.year}-{date.month:02d}"
                    
                    # Add timestamps
                    now = datetime.now(timezone.utc)
                    data['created_at'] = now
                    data['updated_at'] = now
                    data['deleted_at'] = None
                    
                    # Check if record already exists
                    existing = None
                    if model_class == Like and 'link' in data:
                        existing = session.query(model_class).filter_by(link=data['link']).first()
                    elif model_class == Comment and 'comment_id' in data:
                        existing = session.query(model_class).filter_by(comment_id=data['comment_id']).first()
                    elif model_class == Message and 'message_id' in data:
                        existing = session.query(model_class).filter_by(message_id=data['message_id']).first()
                    elif model_class == Connection and 'url' in data:
                        existing = session.query(model_class).filter_by(url=data['url']).first()
                    elif model_class == Post and 'post_link' in data:
                        existing = session.query(model_class).filter_by(post_link=data['post_link']).first()
                    
                    if existing:
                        # Skip existing record
                        batch_skipped += 1
                        continue
                    
                    # Create new record
                    obj = model_class(**data)
                    session.add(obj)
                    batch_records += 1
                    
                except Exception as e:
                    logging.error(f"Error processing row: {str(e)}")
                    logging.error(f"Data that caused error: {data}")
                    batch_errors += 1
                    continue
            
            try:
                session.commit()
                total_records += batch_records
                total_errors += batch_errors
                total_skipped += batch_skipped
                
                # Log batch progress
                batch_time = time.time() - batch_time
                logging.info(f"Processed batch of {batch_size} records in {batch_time:.2f} seconds")
                logging.info(f"Records added: {batch_records}, Errors: {batch_errors}, Skipped: {batch_skipped}")
                batch_time = time.time()
                
            except Exception as e:
                session.rollback()
                logging.error(f"Error committing batch: {str(e)}")
                total_errors += batch_size - batch_records - batch_skipped
        
        logging.info(f"Finished processing {file_path}")
        logging.info(f"Total records added: {total_records}")
        logging.info(f"Total errors: {total_errors}")
        logging.info(f"Total skipped: {total_skipped}")
        
    except Exception as e:
        logging.error(f"Error processing dataset {file_path}: {str(e)}")
        raise
    finally:
        session.close()

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
            'column_mapping': {
                'message_id': 'message_id',
                'sender': 'sender',
                'recipient': 'recipient',
                'message_date': 'message_date',
                'subject': 'subject',
                'content': 'content'
            },
            'date_columns': ['message_date'],
            'skip_rows': 0
        },
        'likes': {
            'file': 'Reactions_cleaned.csv',
            'model': Like,
            'column_mapping': {
                'Date': 'date',
                'Type': 'type',
                'Link': 'link'
            },
            'date_columns': ['date'],
            'skip_rows': 0
        },
        'comments': {
            'file': 'Comments_cleaned.csv',
            'model': Comment,
            'column_mapping': {
                'Date': 'date',
                'Message': 'message',
                'comment_id': 'comment_id',
                'Link': 'link'
            },
            'date_columns': ['date'],
            'skip_rows': 0
        },
        'posts': {
            'file': 'Shares.csv',  # Keep original filename for input
            'model': Post,
            'column_mapping': {
                'Date': 'date',
                'ShareLink': 'post_link',
                'ShareCommentary': 'post_commentary',
                'Visibility': 'visibility'
            },
            'date_columns': ['date'],
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
