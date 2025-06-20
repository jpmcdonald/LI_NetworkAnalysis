-- Drop existing tables if they exist
DROP TABLE IF EXISTS connections;
DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS comments;
DROP TABLE IF EXISTS posts;
DROP TABLE IF EXISTS likes;
DROP TABLE IF EXISTS shares;

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
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP -- Soft delete timestamp
);

-- Create messages table
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(100) UNIQUE,
    sender VARCHAR(200),
    recipient TEXT,
    message_date TIMESTAMP,
    subject TEXT,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP
);

-- Create comments table
CREATE TABLE IF NOT EXISTS comments (
    id SERIAL PRIMARY KEY,
    comment_id VARCHAR(100) UNIQUE,
    date TIMESTAMP,
    message TEXT,
    year INTEGER,
    month INTEGER,
    year_month VARCHAR(7),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP
);

-- Create posts table
CREATE TABLE IF NOT EXISTS posts (
    id SERIAL PRIMARY KEY,
    date TIMESTAMP,
    post_link VARCHAR(300) UNIQUE,
    post_commentary TEXT,
    visibility VARCHAR(50),
    year INTEGER,
    month INTEGER,
    year_month VARCHAR(7),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP
);

-- Create likes table
CREATE TABLE IF NOT EXISTS likes (
    id SERIAL PRIMARY KEY,
    date TIMESTAMP,
    type VARCHAR(50),
    link VARCHAR(300) UNIQUE,
    year INTEGER,
    month INTEGER,
    year_month VARCHAR(7),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP
);

-- Create indexes for connections table
CREATE INDEX IF NOT EXISTS idx_connections_company ON connections(company);
CREATE INDEX IF NOT EXISTS idx_connections_position ON connections(position);
CREATE INDEX IF NOT EXISTS idx_connections_connected_on ON connections(connected_on);
CREATE INDEX IF NOT EXISTS idx_connections_email_address ON connections(email_address);
CREATE INDEX IF NOT EXISTS idx_connections_deleted_at ON connections(deleted_at);

-- Create indexes for messages table
CREATE INDEX IF NOT EXISTS idx_messages_message_id ON messages(message_id);
CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender);
CREATE INDEX IF NOT EXISTS idx_messages_message_date ON messages(message_date);
CREATE INDEX IF NOT EXISTS idx_messages_deleted_at ON messages(deleted_at);

-- Create indexes for comments table
CREATE INDEX IF NOT EXISTS idx_comments_comment_id ON comments(comment_id);
CREATE INDEX IF NOT EXISTS idx_comments_date ON comments(date);
CREATE INDEX IF NOT EXISTS idx_comments_deleted_at ON comments(deleted_at);

-- Create indexes for posts table
CREATE INDEX IF NOT EXISTS idx_posts_date ON posts(date);
CREATE INDEX IF NOT EXISTS idx_posts_post_link ON posts(post_link);
CREATE INDEX IF NOT EXISTS idx_posts_visibility ON posts(visibility);
CREATE INDEX IF NOT EXISTS idx_posts_year ON posts(year);
CREATE INDEX IF NOT EXISTS idx_posts_month ON posts(month);
CREATE INDEX IF NOT EXISTS idx_posts_year_month ON posts(year_month);
CREATE INDEX IF NOT EXISTS idx_posts_deleted_at ON posts(deleted_at);

-- Create indexes for likes table
CREATE INDEX IF NOT EXISTS idx_likes_date ON likes(date);
CREATE INDEX IF NOT EXISTS idx_likes_link ON likes(link);
CREATE INDEX IF NOT EXISTS idx_likes_type ON likes(type);
CREATE INDEX IF NOT EXISTS idx_likes_year ON likes(year);
CREATE INDEX IF NOT EXISTS idx_likes_month ON likes(month);
CREATE INDEX IF NOT EXISTS idx_likes_year_month ON likes(year_month);
CREATE INDEX IF NOT EXISTS idx_likes_deleted_at ON likes(deleted_at);

-- Create indexes for shares table
CREATE INDEX IF NOT EXISTS idx_shares_date ON shares(date);
CREATE INDEX IF NOT EXISTS idx_shares_share_link ON shares(share_link);
CREATE INDEX IF NOT EXISTS idx_shares_visibility ON shares(visibility);
CREATE INDEX IF NOT EXISTS idx_shares_year ON shares(year);
CREATE INDEX IF NOT EXISTS idx_shares_month ON shares(month);
CREATE INDEX IF NOT EXISTS idx_shares_year_month ON shares(year_month);
CREATE INDEX IF NOT EXISTS idx_shares_deleted_at ON shares(deleted_at);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update updated_at for all tables
CREATE TRIGGER update_connections_updated_at
    BEFORE UPDATE ON connections
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_messages_updated_at
    BEFORE UPDATE ON messages
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_comments_updated_at
    BEFORE UPDATE ON comments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_posts_updated_at
    BEFORE UPDATE ON posts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_likes_updated_at
    BEFORE UPDATE ON likes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_shares_updated_at
    BEFORE UPDATE ON shares
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column(); 