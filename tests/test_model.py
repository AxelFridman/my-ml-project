import pandas as pd
from src.model import load_data, train_model

def test_load_data(tmp_path):
    csv = tmp_path / "mini.csv"
    csv.write_text("feature1,feature2,target\n0.5,2.3,1\n0.6,2.1,0\n")
    df = load_data(str(csv))
    assert 'feature1' in df.columns
    assert 'target' in df.columns

def test_train_model_accuracy():
    df = pd.DataFrame({
        'feature1': [0,1,0,1],
        'feature2': [1,0,1,0],
        'target': [0,1,0,1]
    })
    model, acc = train_model(df, target_col='target')
    assert acc > 0.5
