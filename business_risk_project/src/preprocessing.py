import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def derive_risk(self, df):
        """
        Derives the binary 'Risk' target variable from 'Credit_score'.
        Risk = 1 (High) if Credit_score < 600, else 0 (Low).
        """
        print("Deriving 'Risk' target variable...")
        df['Risk'] = (df['Credit_score'] < 600).astype(int)
        print("Risk distribution:")
        print(df['Risk'].value_counts(normalize=True))
        return df

    def preprocess(self, df):
        """
        Preprocesses the data:
        1. Encodes categorical variables.
        2. Drops irrelevant columns (Credit_score is dropped as it's the source of target).
        3. Splits into Train/Test.
        4. Scales features.
        
        Returns:
            X_train_scaled, X_test_scaled, y_train, y_test
        """
        print("Preprocessing data...")
        
        # Encode Business_size
        if 'Business_size' in df.columns:
            df['Business_size'] = self.encoder.fit_transform(df['Business_size'])
            
        # Define Features (X) and Target (y)
        # Drop Credit_score as encoded in Risk
        X = df.drop(columns=['Credit_score', 'Risk'])
        y = df['Risk']
        
        # Stratified Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scaling
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
