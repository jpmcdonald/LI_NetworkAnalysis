import pytest
import pandas as pd
import os
from sqlalchemy import text
from ingest_connections import engine, main, Connection, Session
import logging

# Setup test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_ingestion.log')
    ]
)

@pytest.fixture
def test_db():
    """Create a test database connection."""
    return engine

def cleanup_test_data():
    """Remove test data from the database."""
    session = Session()
    try:
        # Remove test data (connections from test companies)
        test_companies = ['Test Corp', 'Test Inc']
        deleted = session.query(Connection).filter(
            Connection.company.in_(test_companies)
        ).delete()
        session.commit()
        logging.info(f"Cleaned up {deleted} test records")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
        session.rollback()
    finally:
        session.close()

@pytest.fixture(autouse=True)
def cleanup():
    """Automatically clean up test data after each test."""
    yield
    cleanup_test_data()

def test_database_connection():
    """Test database connection."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar() == 1
        logging.info("Database connection test passed")

def test_ingest_connections(test_db, tmp_path):
    """Test connection ingestion functionality."""
    logging.info("Starting ingestion test")
    
    # Create data directory if it doesn't exist
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create test CSV with 3 header rows as expected by the ingestion script
    header_rows = pd.DataFrame([
        ["Header 1"],
        ["Header 2"],
        ["Header 3"]
    ])
    
    # Create test data with exact column names expected by the script
    test_data = pd.DataFrame({
        'First Name': ['John', 'Jane'],
        'Last Name': ['Doe', 'Smith'],
        'URL': ['https://linkedin.com/in/johndoe', 'https://linkedin.com/in/janesmith'],
        'Email Address': ['john@test.com', 'jane@test.com'],
        'Company': ['Test Corp', 'Test Inc'],
        'Position': ['Developer', 'Manager'],
        'Connected On': ['01 Jan 2023', '02 Jan 2023']
    })
    
    # Save to the expected location
    csv_path = data_dir / "Connections.csv"
    
    # First write the header rows
    with open(csv_path, 'w', newline='') as f:
        header_rows.to_csv(f, index=False, header=False)
    
    # Then append the data with headers
    test_data.to_csv(csv_path, mode='a', index=False)
    
    # Change to the temp directory so the script finds the data directory
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    try:
        # Test ingestion
        main()
        
        # Verify data was ingested
        session = Session()
        try:
            # Check total count
            count = session.query(Connection).count()
            assert count >= 2  # At least our test data should be there
            logging.info(f"Verified {count} total records in database")
            
            # Verify specific records
            john = session.query(Connection).filter_by(url='https://linkedin.com/in/johndoe').first()
            assert john is not None
            assert john.first_name == 'John'
            assert john.last_name == 'Doe'
            assert john.company == 'Test Corp'
            logging.info("Verified John Doe's record")
            
            jane = session.query(Connection).filter_by(url='https://linkedin.com/in/janesmith').first()
            assert jane is not None
            assert jane.first_name == 'Jane'
            assert jane.last_name == 'Smith'
            assert jane.company == 'Test Inc'
            logging.info("Verified Jane Smith's record")
        finally:
            session.close()
    finally:
        # Change back to original directory
        os.chdir(original_dir)
        logging.info("Test completed successfully") 