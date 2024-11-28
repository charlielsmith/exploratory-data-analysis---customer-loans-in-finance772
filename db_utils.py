import yaml
import pandas as pd
from sqlalchemy import create_engine
import psycopg2

# Function to load credentials
def load_credentials(filepath='credentials.yaml'):
    """
    Load database credentials from a YAML file.

    Args:
        filepath (str): The path to the credentials.yaml file.
    
    Returns:
        dict: A dictionary containing the database credentials.
    """
    try:
        with open(filepath, 'r') as file:
            credentials = yaml.safe_load(file)
        return credentials
    except Exception as e:
        print(f"Error loading credentials: {e}")
        return None

# RDSDatabaseConnector class
class RDSDatabaseConnector:
    def __init__(self, credentials):
        """
        Initializes the RDSDatabaseConnector with database credentials.

        Args:
            credentials (dict): A dictionary containing the database connection details.
                                Expected keys: 'host', 'port', 'username', 'password', 'database'.
        """
        self.host = credentials.get('RDS_HOST')
        self.port = credentials.get('RDS_PORT')
        self.username = credentials.get('RDS_USER')
        self.password = credentials.get('RDS_PASSWORD')
        self.database = credentials.get('RDS_DATABASE')

        # Validate that all required credentials are present
        if not all([self.host, self.port, self.username, self.password, self.database]):
            raise ValueError("Missing one or more required database credentials!")

    def init_engine(self):
        """
        Initializes and returns a SQLAlchemy engine using the provided credentials.

        Returns:
            sqlalchemy.engine.Engine: A SQLAlchemy engine object.
        """
        try:
            connection_url = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            engine = create_engine(connection_url)
            print("SQLAlchemy engine initialized successfully!")
            return engine
        except Exception as e:
            print(f"Error initializing the SQLAlchemy engine: {e}")
            return None

    def extract_data(self, engine, table_name='loan_payments'):
        """
        Extracts data from the specified table in the RDS database and returns it as a Pandas DataFrame.

        Args:
            engine (sqlalchemy.engine.Engine): SQLAlchemy engine object for connecting to the database.
            table_name (str): Name of the table to extract data from.

        Returns:
            pandas.DataFrame: DataFrame containing the data from the table.
        """
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, engine)
            print("Data extraction successful!")
            return df
        except Exception as e:
            print(f"Error extracting data: {e}")
            return None

# Function to save data to a CSV file
def save_data_to_csv(data_frame, file_name):
    """
    Saves the provided DataFrame to a CSV file.

    Args:
        data_frame (pd.DataFrame): The data to be saved.
        file_name (str): The name of the file where the data should be saved.

    Returns:
        None
    """
    data_frame.to_csv(file_name, index=False)
    print(f"Data has been successfully saved to {file_name}.")

# Main script
if __name__ == "__main__":
    # Step 1: Load the credentials
    credentials = load_credentials('credentials.yaml')

    # Step 2: Pass the credentials to the RDSDatabaseConnector
    if credentials:
        rds_connector = RDSDatabaseConnector(credentials)

        # Step 3: Initialize the SQLAlchemy engine
        engine = rds_connector.init_engine()

        # Step 4: Extract data from the database
        if engine:
            data_frame = rds_connector.extract_data(engine)

            # Step 5: Save the data to a CSV file
            if data_frame is not None:
                save_data_to_csv(data_frame, "loan_payments.csv")
    



# Create a function which will load the data from your local machine into a Pandas DataFrame
df = pd.read_csv('loan_payments.csv')

# Assuming you've already loaded your credentials and connected to the database
credentials = load_credentials('credentials.yaml')

if credentials:
    rds_connector = RDSDatabaseConnector(credentials)
    engine = rds_connector.init_engine()

    # Extract data
    df = rds_connector.extract_data(engine)

    if df is not None:
        # Save the data to a CSV file in a specific directory
        save_data_to_csv(df, './data/loan_payments.csv')  # Ensure the 'data' folder exists
        # Or specify an absolute path if preferred
        # save_data_to_csv(df, '/path/to/your/directory/loan_payments.csv')

print(f"The shape of the data is: {df.shape}")