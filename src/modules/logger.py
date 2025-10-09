import os
from datetime import datetime


class ExperimentLogger:
    """Logger for experiment results

    Logs to both console and file simultaneously.
    """

    def __init__(self, experiment_name, results_dir='results'):
        """
        Args:
            experiment_name: Name of the experiment (used in log filename)
            results_dir: Directory to save log files (default: 'results')
        """
        self.experiment_name = experiment_name

        # Create results directory
        os.makedirs(results_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(results_dir, f'{experiment_name}_{timestamp}.log')

        # Write header
        self.log(f'=== Experiment: {experiment_name} ===')
        self.log(f'Started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.log('')

    def log(self, message):
        """Log message to both console and file

        Args:
            message: Message to log
        """
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def log_config(self, config_dict):
        """Log configuration dictionary

        Args:
            config_dict: Dictionary of configuration parameters
        """
        self.log('Configuration:')
        for key, value in config_dict.items():
            self.log(f'  {key}: {value}')
        self.log('')

    def log_metrics(self, epoch, metrics_dict):
        """Log metrics for an epoch

        Args:
            epoch: Current epoch number
            metrics_dict: Dictionary of metrics
        """
        metrics_str = ' '.join([f'{k}: {v:.4f}' for k, v in metrics_dict.items()])
        self.log(f'Epoch [{epoch}] {metrics_str}')

    def close(self):
        """Close logger and write footer"""
        self.log('')
        self.log(f'Finished at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.log(f'Log saved to: {self.log_file}')
