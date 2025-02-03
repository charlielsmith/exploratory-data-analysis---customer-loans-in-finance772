Description

This project involves analysing customer loan payment data and connecting to a cloud-based RDS (Relational Database Service) using Python. The project involves:
	•	Loading credentials securely from a YAML file.
	•	Connecting to an RDS database with SQLAlchemy.
	•	Extracting data into a Pandas DataFrame for further analysis.
	•	Data preprocessing and analysis: handling missing values, skewness analysis, detecting collinearity with VIF.
	•	Conducting data analysis to understand loan payment trends.


Aim of the Project
The primary objective is to combine database management and Python data analysis skills. This includes securely handling credentials, performing data extraction using SQL, and analysing real-world financial data.

Learned
	•	Used YAML files for secure credential management.
	•	Worked with SQLAlchemy for database connections.
	•	Pandas DataFrame operations for analsing datasets.
	•	Handling missing values, imputation, skewness analysis, detecting collinearity with VIF.

Installation instructions
	•	Clone repository: git clone https://github.com/charlielsmith/exploratory-data-analysis---customer-loans-in-finance772.git
 cd your-repo-name
	•	Install: pip install pandas sqlalchemy psycopg2 pyyaml
	•	Set up credentials

 Usage instructions
 	•	Run project: python db_utils.py
   	•	Data analysis: print(df.info)## Usage Instructions
	•   Install dependencies: pip install -r requirements.txt

	•   Ensure credentials are set up by creating a credentials.yaml file with database credentials

	•   Run the main script to connect to the database and load the dataset
	        python db_utils.py

    •   Open a Python script or Jupyter Notebook and import necessary modules
            from data_cleaning import DataTransform
            from data_analysis import DataFrameTransform
            import pandas as pd

    •   Load the dataset
            df = pd.read_csv('loan_payments.csv')

    •   Create a transformation object
            df_transform = DataFrameTransform(df)

    •   Drop rows with missing funded_amount values
            df_transform.drop_rows_with_missing_funded_amount()

    •   Drop highly correlated columns (loan_amount and funded_amount_inv)
            df_transform.drop_highly_correlated()

    •   Retrieve the cleaned DataFrame
            df_cleaned = df_transform.get_dataframe()

    •   Display the cleaned DataFrame
            print(df_cleaned.head())


File structure

project-root/
│── credentials.yaml      # Database credentials file (not in repo, ignored in .gitignore)
│── db_utils.py           # Main Python script for RDS connection and data extraction
│── data_cleaning.py      # Module for handling missing values and feature selection
│── data_analysis.py      # Module for calculating VIF and analyzing data
│── loan_payments.csv     # Sample dataset for analysis
│── README.md             # Project documentation
│── requirements.txt      # List of Python dependencies


License  

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  