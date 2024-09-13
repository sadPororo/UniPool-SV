""" Main function to initiate the experiment """
from train import train_main
from evaluate import evaluate_main
from utils.parser import get_config, load_eval_config

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    config = get_config()
    
    if config['posarg']['action'] == 'train':
        train_main(config)
                
    elif config['posarg']['action'] == 'eval':
        config = load_eval_config(config)
        evaluate_main(config)
                
