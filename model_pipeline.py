import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def create_and_save_pipeline(df_train):
    """
    Создает и сохраняет пайплайн для предобработки и моделирования
    """
    # Подготовка данных
    features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    
    # Оставляем только существующие колонки
    features = [col for col in features if col in df_train.columns]
    
    X_train = df_train[features].copy()
    y_train = df_train['selling_price'].copy()
    
    # Создаем пайплайн
    numeric_features = features
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    
    # Создаем полный пайплайн
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    # Обучаем пайплайн
    pipeline.fit(X_train, y_train)
    
    # Сохраняем пайплайн
    joblib.dump(pipeline, 'model_pipeline.pkl')
    
    # Сохраняем список признаков
    joblib.dump(features, 'model_features.pkl')
    
    return pipeline, features

def load_pipeline():
    """
    Загружает сохраненный пайплайн
    """
    pipeline = joblib.load('model_pipeline.pkl')
    features = joblib.load('model_features.pkl')
    return pipeline, features