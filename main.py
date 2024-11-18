import matplotlib.pyplot as plt
import numpy as np
from args import get_args
from runners.clean_up_ppo_runner import CleanupPPORunner

args = get_args()
runner = CleanupPPORunner(args)
runner.train()
runner.data_collector.save_to_csv()
