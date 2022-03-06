import pathlib as pl

this_file = pl.Path(__file__)
root = this_file.parents[2].resolve()

data = root / 'data'
data_raw = data / 'raw'
data_interim = data / 'interim'
data_processed = data / 'processed'
data_annotations = data / 'data.csv'

models = root / 'models'
model_predictions = models / 'predictions'

should_exist = [
    data_interim,
    data_processed,
    models,
    model_predictions
]

for p in should_exist:
    p.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    print(f"files.py module: {this_file}")
    print(f"project root: {root}")

    print(f"data directory: {data}")
    print(f"annotations file: {data_annotations}")

    print(f"models directory: {models}")
    print(f"model predictions directory: {model_predictions}")