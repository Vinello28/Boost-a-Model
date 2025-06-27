import logging
import time


def timelog(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logging.debug(f"{func.__name__} took {end - start:.6f} seconds")
        return result

    return wrapper
