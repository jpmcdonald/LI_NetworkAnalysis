try:
    with open('data/Connections.csv', 'r', encoding='utf-8') as file:
        print("First 10 lines of the file:")
        print("-" * 50)
        for i, line in enumerate(file):
            if i < 10:  # Print the first 10 lines
                print(f"Line {i+1}: {line.strip()}")
            else:
                break
except UnicodeDecodeError:
    # If UTF-8 fails, try with a different encoding
    with open('data/Connections.csv', 'r', encoding='latin-1') as file:
        print("First 10 lines of the file (using latin-1 encoding):")
        print("-" * 50)
        for i, line in enumerate(file):
            if i < 10:  # Print the first 10 lines
                print(f"Line {i+1}: {line.strip()}")
            else:
                break
except Exception as e:
    print(f"Error reading file: {str(e)}") 