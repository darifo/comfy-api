from queue import Queue
from concurrent.futures import Future
import threading
import time
import logging

requests_queue = Queue()

def task_worker():
    while True:
        task, future, params = requests_queue.get()
        # logging.info(f"task_worker get task: {threading.current_thread().name}")
        try:
            rt = task(params)
            # result = f"task finished: {task}"
            future.set_result(rt)
        except Exception as e:
            logging.error(f"Error in task: {e}")
        finally:
            requests_queue.task_done()

for i in range(5):
    threading.Thread(target=task_worker, daemon=True, name=f'thread_{str(i)}').start()

# threading.Thread(target=task_worker, daemon=True).start()

def task(params):
    logging.info(f"task start: {params}")
    time.sleep(1)
    return params

def main():

    # logging.info("main start")
    # logging.info(f"main thread: {threading.current_thread().name}")

    futures = []
    for i in range(10):
        future = Future()
        # 传输
        requests_queue.put((task, future, (1, 2, 3)))
        futures.append(future)
        logging.info(f"main put task: {i}")
    
    time.sleep(10)