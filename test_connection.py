from ingest_connections import engine
from sqlalchemy import text

def test_connection():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("Successfully connected to the database!")
            return True
    except Exception as e:
        print(f"Failed to connect to the database: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection() 