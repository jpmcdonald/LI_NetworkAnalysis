import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Date, Text, TIMESTAMP, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime, UTC
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
    
# Setup logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('linkedin_ingestion.log')
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
    __table_args__ = (UniqueConstraint('url', name='uix_url'),)

# Create engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%d %b %Y').date()
    except Exception:
        return None

def main():
    batch_time = datetime.now(UTC)
    logging.info(f'Starting Connections ingestion at {batch_time}')
    
    try:
        df = pd.read_csv('data/Connections.csv', encoding='utf-8', skiprows=3)
        logging.info(f'Successfully read CSV file with {len(df)} rows')
    except Exception as e:
        logging.error(f'Error reading CSV: {e}')
        return

    # Rename columns to match DB
    df = df.rename(columns={
        'First Name': 'first_name',
        'Last Name': 'last_name',
        'URL': 'url',
        'Email Address': 'email_address',
        'Company': 'company',
        'Position': 'position',
        'Connected On': 'connected_on',
    })
    logging.info('Columns renamed to match database schema')

    # Parse dates
    df['connected_on'] = df['connected_on'].apply(parse_date)
    df['cdc_datetime'] = batch_time
    logging.info('Dates parsed and CDC timestamp added')

    # Fill missing columns if any
    for col in ['notes']:
        if col not in df.columns:
            df[col] = None

    # Process in batches
    batch_size = 1000
    total_inserted = 0
    total_skipped = 0
    total_rows = len(df)
    
    for batch_start in range(0, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        batch = df.iloc[batch_start:batch_end]
        batch_time = datetime.now(UTC)
        
        logging.info(f'Processing batch {batch_start//batch_size + 1} of {(total_rows + batch_size - 1)//batch_size} (rows {batch_start+1}-{batch_end})')
        
        session = Session()
        inserted, skipped = 0, 0
        new_connections = []
        
        try:
            for _, row in batch.iterrows():
                data = row.to_dict()
                stmt = insert(Connection).values(**data)
                stmt = stmt.on_conflict_do_nothing(index_elements=['url'])
                try:
                    result = session.execute(stmt)
                    if result.rowcount:
                        inserted += 1
                        new_connections.append(f"{data['first_name']} {data['last_name']} ({data['company']})")
                    else:
                        skipped += 1
                except Exception as e:
                    logging.error(f'Error inserting row: {e} | Data: {data}')
            
            session.commit()
            
            # Update totals
            total_inserted += inserted
            total_skipped += skipped
            
            # Log batch summary
            logging.info(f'Batch {batch_start//batch_size + 1} complete at {batch_time}')
            logging.info(f'Batch rows processed: {len(batch)}')
            logging.info(f'Batch new connections: {inserted}')
            logging.info(f'Batch skipped: {skipped}')
            if new_connections:
                logging.info('New connections in this batch:')
                for conn in new_connections:
                    logging.info(f'  - {conn}')
        
        except Exception as e:
            logging.error(f'Error processing batch: {e}')
            session.rollback()
        finally:
            session.close()
    
    # Log final summary
    end_time = datetime.now(UTC)
    duration = end_time - batch_time
    logging.info('=== Ingestion Complete ===')
    logging.info(f'Total time: {duration}')
    logging.info(f'Total rows processed: {total_rows}')
    logging.info(f'Total new connections: {total_inserted}')
    logging.info(f'Total skipped: {total_skipped}')
    logging.info(f'Success rate: {(total_inserted + total_skipped) / total_rows * 100:.2f}%')

if __name__ == '__main__':
    main() 