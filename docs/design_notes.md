# LinkedIn Connections Data Analysis Project

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
├── docs/               # Documentation
│   ├── design_notes.md # This file
│   └── README.md      # Project overview
└── data/              # Data directory
    └── connections.csv # LinkedIn connections data
```

## Setup Instructions

1. Create a `.env` file with the following variables:
   ```
   POSTGRES_USER=linkedin_user
   POSTGRES_PASSWORD=linkedin_password
   POSTGRES_DB=linkedin_db
   PGADMIN_EMAIL=admin@example.com
   PGADMIN_PASSWORD=admin_password
   ```

2. Start the services:
   ```bash
   docker-compose up -d
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your LinkedIn connections CSV file in the `data/` directory as `connections.csv`

5. Run the ingestion script:
   ```bash
   python ingest_connections.py
   ```

## Database Schema

The `connections` table stores LinkedIn connection data with the following columns:
- id (SERIAL PRIMARY KEY)
- first_name (VARCHAR(255))
- last_name (VARCHAR(255))
- company (VARCHAR(255))
- position (VARCHAR(255))
- connected_date (DATE)
- email (VARCHAR(255))
- phone (VARCHAR(20))
- notes (TEXT)
- created_at (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
- updated_at (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)

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