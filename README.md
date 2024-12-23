Description

This project involves analysing customer loan payment data and connecting to a cloud-based RDS (Relational Database Service) using Python. The project involves:
	•	Loading credentials securely from a YAML file.
	•	Connecting to an RDS database with SQLAlchemy.
	•	Extracting data into a Pandas DataFrame for further analysis.
	•	Conducting data analysis to understand loan payment trends.


Aim of the Project
The primary objective is to combine database management and Python data analysis skills. This includes securely handling credentials, performing data extraction using SQL, and analysing real-world financial data.

Learned
	•	Used YAML files for secure credential management.
	•	Worked with SQLAlchemy for database connections.
	•	Pandas DataFrame operations for analsing datasets.

Installation instructions
	•	Clone repository: git clone https://github.com/your-username/your-repo-name.git
 cd your-repo-name
	•	Install: pip install pandas sqlalchemy psycopg2 pyyaml
	•	Set up credentials

 Usage instructions
 		•	Run project: python db_utils.py
   	•	Data analysis: print(df.info)


File structure

project-root/
│
├── credentials.yaml       # Database credentials file (not in the repo but in a .gitignore file)
├── db_utils.py            # Main Python script for RDS connection and data extraction
├── loan_payments.csv      # Sample dataset for analysis
├── README.md              # Project documentation
└── requirements.txt       # List of Python dependencies
