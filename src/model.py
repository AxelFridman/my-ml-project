import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(path):
    df = pd.read_csv(path)
    return df

def train_model(df, target_col='target'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return clf, acc

if __name__ == '__main__':
    df = load_data('data/sample_data.csv')
    model, accuracy = train_model(df)
    print(f'Model trained with accuracy: {accuracy:.4f}')
