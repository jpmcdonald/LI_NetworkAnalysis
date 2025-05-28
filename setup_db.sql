-- Drop existing table if it exists
DROP TABLE IF EXISTS connections;

-- Create connections table
CREATE TABLE IF NOT EXISTS connections (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    url VARCHAR(300) UNIQUE,
    email_address VARCHAR(200),
    company VARCHAR(200),
    position VARCHAR(200),
    connected_on DATE,
    cdc_datetime TIMESTAMP NOT NULL, -- CDC batch load timestamp
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on commonly searched fields
CREATE INDEX IF NOT EXISTS idx_connections_company ON connections(company);
CREATE INDEX IF NOT EXISTS idx_connections_position ON connections(position);
CREATE INDEX IF NOT EXISTS idx_connections_connected_on ON connections(connected_on);
CREATE INDEX IF NOT EXISTS idx_connections_email_address ON connections(email_address);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_connections_updated_at
    BEFORE UPDATE ON connections
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column(); 