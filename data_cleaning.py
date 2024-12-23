# data_cleaning.py

import pandas as pd
from datetime import datetime

class DataTransform:
    def __init__(self, df):
        self.df = df

    def convert_sub_grade(self):

        """Convert sub_grade column from categorical to numerical."""

        self.df['sub_grade'] = self.df['sub_grade'].map({
            'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'A5': 5, 
            'B1': 6, 'B2': 7, 'B3': 8, 'B4': 9, 'B5': 10, 
            'C1': 11, 'C2': 12, 'C3': 13, 'C4': 14, 'C5': 15, 
            'D1': 16, 'D2': 17, 'D3': 18, 'D4': 19, 'D5': 20,
            'E1': 21, 'E2': 22, 'E3': 23, 'E4': 24, 'E5': 25, 
            'F1': 26, 'F2': 27, 'F3': 28, 'F4': 29, 'F5': 30, 
            'G1': 31, 'G2': 32, 'G3': 33, 'G4': 34, 'G5': 35
        })
        return self.df
    
    def convert_grade(self):
        """Convert grade column from categorical to numerical."""
        grade_map = {
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7
        }
        # Strip whitespace and convert grades
        self.df['grade'] = self.df['grade'].astype(str).str.strip()

        # Map values
        self.df['grade'] = self.df['grade'].map(grade_map)
        
        # Print the result of the mapping to check if it's correct
        print(self.df['grade'].head())  # Print the first few rows to inspect
        
        # If there are NaN values after mapping, you can fill them or handle them
        if self.df['grade'].isna().any():
            print("Warning: Some grades were not mapped correctly.")
            self.df['grade'] = self.df['grade'].fillna(-1)  # Fill NaN with -1 or another placeholder
        
        return self.df

    def convert_employment_length(self):

        """Converts employment_length column into numerical values."""
 
        self.df['employment_length'] = self.df['employment_length'].replace({
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
            '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
            '8 years': 8, '9 years': 9, '10+ years': 10
        })
        return self.df

    def convert_home_ownership(self):

        """Converts home_ownership into numerical values."""

        self.df['home_ownership'] = self.df['home_ownership'].map({
            'MORTGAGE': 1, 'RENT': 2
        })
        return self.df

    def convert_issue_date(self):
        """Converts issue_date to datetime format."""
        self.df['issue_date'] = pd.to_datetime(self.df['issue_date'], errors='coerce')
        return self.df

    def convert_last_payment_date(self):
        """Converts last_payment_date to datetime format."""
        self.df['last_payment_date'] = pd.to_datetime(self.df['last_payment_date'], errors='coerce')

        current_date = datetime.now()
        self.df['last_payment_date'] = (current_date - self.df['last_payment_date']).dt.days / 365.25
        return self.df

    def convert_last_credit_pull_date(self):
        """Converts last_credit_pull_date to datetime format."""
        self.df['last_credit_pull_date'] = pd.to_datetime(self.df['last_credit_pull_date'], errors='coerce')
        return self.df
    
    def convert_term(self):
        """Creates a mapping dictionary and converts values"""
        # Handling missing values and converting '36 months' and '60 months'
        self.df['term'] = self.df['term'].replace({None: 'None', '36 months': 1, '60 months': 2})

        # If there are still 'None' values, you can map them to 0, or you can keep them as NaN
        self.df['term'] = self.df['term'].map({'None': 0, 1: 1, 2: 2})
        return self.df
    
    def convert_verification_status(self):
        """Convert verification_status column to numerical values."""
        self.df['verification_status'] = self.df['verification_status'].map({
            'Not Verified': 0,
            'Verified': 1,
            'Source Verified': 2
        })
        return self.df
    
    def convert_earliest_credit_line(self):
        from datetime import datetime
        self.df['earliest_credit_line'] = pd.to_datetime(
            self.df['earliest_credit_line'], format='%b-%Y', errors='coerce'
        )
        current_date = datetime.now()
        self.df['credit_age_years'] = (current_date - self.df['earliest_credit_line']).dt.days / 365.25
        return self.df
    
# Example usage
data_frame = pd.read_csv('loan_payments.csv')  # Assuming you're loading data
transformer = DataTransform(data_frame)

# Applying transformations
data_frame = transformer.convert_sub_grade()
data_frame = transformer.convert_employment_length()
data_frame = transformer.convert_home_ownership()
data_frame = transformer.convert_issue_date()
data_frame = transformer.convert_last_payment_date()
data_frame = transformer.convert_last_credit_pull_date()
data_frame = transformer.convert_term()

# Clean loan_status column
data_frame['loan_status'] = data_frame['loan_status'].str.extract(r'(Fully Paid|Current|Charged Off|Late|Does not meet the credit policy)').fillna('Unknown')
data_frame['loan_status'] = data_frame['loan_status'].astype('category')

# Print categories
print(data_frame['loan_status'].cat.categories)

# Map loan_status to numeric codes
data_frame['loan_status'] = data_frame['loan_status'].cat.codes

data_frame = transformer.convert_grade()  # Make sure this line is included

data_frame = transformer.convert_earliest_credit_line()