from src.config import files


class Config:
    def __init__(self):
        self.epochs = 20
        self.num_classes = 9
        self.batch_size = 32
        self.learning_rate = 0.01
        self.dataset = "CellPainting"
        self.architecture = "CNN"
        self.pin_memory = False
        self.momentum = 0.9
        self.step_size = 3
        self.gamma = 0.1
        self.dataset_metadata = files.data_annotations
        self.num_workers = 0
        self.data_root_dir = files.data_processed
        self.model_path = files.models / 'model.pth'
        self.model_predictions = files.model_predictions / 'results.csv'