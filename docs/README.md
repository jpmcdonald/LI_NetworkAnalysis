# LinkedIn Data Analysis Project

This project provides tools for analyzing LinkedIn connection data, including data ingestion, storage, and analysis capabilities.

## Project Structure

```
your_project/
│
├── docker-compose.yml    # Docker services configuration
├── setup_db.sql         # Database initialization script
├── .env                 # Environment variables
├── requirements.txt     # Python dependencies
├── ingest_connections.py # Main data ingestion script
├── test_ingest_connections.py # Test suite
├── test_connections.py # Test connection to the postgres server
├── docs/               # Documentation
│   └── README.md      # Project overview
└── data/              # Data directory
    └── connections.csv # LinkedIn connections data
```

## Setup Instructions

1. **Environment Setup**
   - Create a `.env` file with the following variables:
     ```
     POSTGRES_USER=your_user
     POSTGRES_PASSWORD=your_password
     POSTGRES_DB=linkedin_db
     POSTGRES_HOST=localhost
     POSTGRES_PORT=5433
     ```

2. **Database Setup**
   - Start the PostgreSQL database using Docker:
     ```bash
     docker-compose up -d
     ```
   - The database schema will be automatically created using `setup_db.sql`

3. **Python Environment**
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

## Database Schema (`setup_db.sql`)

The `setup_db.sql` file sets up the database schema with the following features:

1. **Connections Table**
   - Stores LinkedIn connection data
   - Includes fields for name, company, position, etc.
   - Uses CDC (Change Data Capture) for tracking data updates

2. **Indexes**
   - Optimized for common queries
   - Includes indexes on company, position, and connection date

3. **Automatic Timestamps**
   - `created_at`: When the record was first inserted
   - `updated_at`: Automatically updated when record changes
   - `cdc_datetime`: Batch load timestamp for tracking imports

## Usage

1. **Data Ingestion**
   ```bash
   python ingest_connections.py
   ```
   - Reads LinkedIn Connections CSV
   - Handles duplicate detection
   - Logs all operations

2. **Data Analysis**
   Example queries for analyzing your network:
   ```sql
   -- Count connections by company
   SELECT company, COUNT(*) as connection_count 
   FROM connections 
   GROUP BY company 
   ORDER BY connection_count DESC;

   -- Track network growth
   SELECT 
       DATE_TRUNC('month', connected_on) as month,
       COUNT(*) as new_connections
   FROM connections 
   GROUP BY DATE_TRUNC('month', connected_on)
   ORDER BY month DESC;
   ```

## Testing

Run the test suite with:
```bash
pytest test_ingest_connections.py
```

To run a specific test:
```bash
pytest test_ingest_connections.py::test_function_name
```

## Accessing pgAdmin

1. Open http://localhost:5050 in your browser
2. Login with the credentials from your .env file
3. Add a new server connection:
   - Host: postgres
   - Port: 5432
   - Database: linkedin_db
   - Username: linkedin_user
   - Password: linkedin_password

## Security

- Database credentials are stored in `.env` file (not in version control)
- CDC timestamps track all data changes
- Unique constraints prevent duplicate entries
- Use strong passwords and keep them secure
- Regularly update dependencies to patch security vulnerabilities

## Development

- Use `pytest` for running tests
- Follow PEP 8 style guide
- Use `black` for code formatting
- Write clear and concise code
- Document your code with comments

## Logging

- All operations are logged to `linkedin_ingestion.log`
- Includes detailed information about data processing
- Tracks new and existing connections
- Use `tail -f linkedin_ingestion.log` to view logs in real-time 