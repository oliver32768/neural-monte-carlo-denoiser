import logging
from train_utils import log_to_file

class EarlyStopper:
    def __init__(self, log_filepath, patience=1, min_delta=0):
        self.log_filepath = log_filepath
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0

            counter_info = f'Early stopping counter reset: {self.counter}/{self.patience}'
            logging.info(counter_info)
            log_to_file(self.log_filepath, counter_info)

            return False
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1

            counter_info = f'Early stopping counter incremented: {self.counter}/{self.patience}'
            logging.info(counter_info)
            log_to_file(self.log_filepath, counter_info)
            
            return self.counter >= self.patience

        counter_info = f'Early stopping counter sustained: {self.counter}/{self.patience}'
        logging.info(counter_info)
        log_to_file(self.log_filepath, counter_info)

        return False
