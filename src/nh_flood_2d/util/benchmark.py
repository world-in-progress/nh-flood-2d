import time

def benchmark(applied: bool):
    """
    A simple benchmarking decorator to measure the execution time of a function.
    If `applied` is True, it will print the execution time; otherwise, it will do nothing.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if applied:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                print(f'Execution time of {func.__name__}: {end_time - start_time:.4f} seconds')
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator