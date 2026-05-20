# src/utils.py
import os
import logging


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def setup_logging(level=logging.INFO):
    """
    Call once from main.py. All modules use logging.getLogger(__name__)
    and inherit this configuration automatically.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(RESULTS_DIR, "run.log"), mode="w")
        ]
    )
