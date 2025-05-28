import pandas as pd
import csv

print("Analyzing LinkedIn Connections CSV file...")
print("-" * 50)

# Try reading with pandas
try:
    print("\nReading with pandas:")
    # Skip the first 3 rows to reach the actual header
    df = pd.read_csv('data/Connections.csv', encoding='utf-8', skiprows=3)
    
    # Basic statistics
    total_connections = len(df)
    connections_with_email = df['Email Address'].notna().sum()
    email_percentage = (connections_with_email / total_connections) * 100
    
    print(f"\nTotal connections: {total_connections}")
    print(f"Connections with email addresses: {connections_with_email}")
    print(f"Percentage with email addresses: {email_percentage:.1f}%")
    
    # Show first 5 rows
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    
    # Show column names and data types
    print("\nColumn information:")
    print(df.dtypes)
    
except Exception as e:
    print(f"Pandas error: {str(e)}")

# Try reading with csv module as backup
try:
    print("\nTrying csv module (first 5 rows):")
    with open('data/Connections.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Get headers
        print("\nHeaders:", headers)
        for i, row in enumerate(reader):
            if i < 5:
                print(f"Row {i+1}: {row}")
            else:
                break
except Exception as e:
    print(f"CSV module error: {str(e)}")

# Try reading as plain text
try:
    print("\nTrying plain text read:")
    with open('data/Connections.csv', 'r', encoding='utf-8') as file:
        print("\nFirst 5 lines:")
        for i, line in enumerate(file):
            if i < 5:
                print(f"Line {i+1}: {line.strip()}")
            else:
                break
except Exception as e:
    print(f"Plain text error: {str(e)}")

print("\nScript execution complete.") 