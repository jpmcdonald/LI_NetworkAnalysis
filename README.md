# LinkedIn Connection Data Analysis

This project provides tools for analyzing LinkedIn connection data by storing it in a PostgreSQL database and providing analysis capabilities.

## Setup

1. Install Docker if you haven't already
2. Clone this repository
3. Run the setup script:
```bash
./setup.sh
```

## Code Quality

This project uses Black and Flake8 to maintain code quality:

### Black (Code Formatter)
Black automatically formats your Python code to follow a consistent style. To format your code:
```bash
black .
```

### Flake8 (Code Linter)
Flake8 checks your code for style and programming errors. To lint your code:
```bash
flake8 .
```

### Pre-commit Hooks
To automatically run these tools before each commit, install pre-commit:
```bash
pip install pre-commit
pre-commit install
```

## Data Ingestion

The `ingest_connections.py` script processes LinkedIn connection data from CSV files and stores it in PostgreSQL.

### Running the Ingestion Script

1. Export your LinkedIn connections as CSV from LinkedIn
2. Place the CSV file in the `data` directory
3. Run the ingestion script:
```bash
python ingest_connections.py
```

### Logging

The ingestion script includes comprehensive logging:
- Logs are written to both console and `ingestion.log`
- Log level is set to INFO by default
- Logs include:
  - File processing status
  - Record counts
  - Database operations
  - Any errors or warnings

## Testing

The project includes automated tests to ensure data integrity and functionality:

### Running Tests

```bash
pytest test_ingest_connections.py -v
```

### Test Features

- Tests run in an isolated environment
- Test data is automatically cleaned up after each test
- Tests verify:
  - Database connectivity
  - Data ingestion
  - Data integrity
  - Error handling

### Test Logging

- Test execution is logged to `test_ingestion.log`
- Logs include:
  - Test setup and teardown
  - Data verification steps
  - Cleanup operations
  - Any test failures or errors

## Data Analysis

The `analyze_connections.py` script provides various analysis capabilities:

### Running Analysis

```bash
python analyze_connections.py
```

### Available Analyses

1. Connection Growth Over Time
   - Monthly connection growth
   - Cumulative connections
   - Growth rate trends

2. Company Analysis
   - Top companies by connection count
   - Company size distribution
   - Industry trends

3. Position Analysis
   - Common job titles
   - Title trends over time
   - Seniority level distribution

4. Geographic Analysis
   - Connection distribution by location
   - Regional trends
   - Global connection map

## Database Schema

The PostgreSQL database uses the following schema:

```sql
CREATE TABLE connections (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    url VARCHAR(255) UNIQUE,
    email VARCHAR(255),
    company VARCHAR(255),
    position VARCHAR(255),
    connected_on DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Example Queries

### Connection Growth
```sql
SELECT 
    DATE_TRUNC('month', connected_on) as month,
    COUNT(*) as new_connections,
    SUM(COUNT(*)) OVER (ORDER BY DATE_TRUNC('month', connected_on)) as total_connections
FROM connections
GROUP BY month
ORDER BY month;
```

### Top Companies
```sql
SELECT 
    company,
    COUNT(*) as connection_count
FROM connections
WHERE company IS NOT NULL
GROUP BY company
ORDER BY connection_count DESC
LIMIT 10;
```

### Position Analysis
```sql
SELECT 
    position,
    COUNT(*) as count
FROM connections
WHERE position IS NOT NULL
GROUP BY position
ORDER BY count DESC
LIMIT 20;
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 