import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def train(args):
    input_data_path = os.path.join('/opt/ml/input/data/train')
    train_data = pd.read_csv(os.path.join(input_data_path, 'train_data.csv'))
    
    X = train_data.drop('target', axis=1)
    y = train_data['target']

    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    continuous_columns = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    scaler = StandardScaler()
    X[continuous_columns] = scaler.fit_transform(X[continuous_columns])

    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
    model.fit(X, y)

    model_dir = os.path.join('/opt/ml/model')
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=None)

    args = parser.parse_args()
    train(args)
