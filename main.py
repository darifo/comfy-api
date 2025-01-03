import logging
import os
from comfy_api.cli_args import args
from comfy_api.logger import setup_logger

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["PYTHONIOENCODING"] = "utf-8"

setup_logger(log_level="INFO", use_stdout=True)

from fastapi import FastAPI, BackgroundTasks
from concurrent.futures import ThreadPoolExecutor, Future
import time

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=5)

def long_task(message):
    time.sleep(5)
    return f"Hello, {message}! (after 5s)"

@app.post("/async_task")
async def create_task(message: str, background_tasks: BackgroundTasks):
    future: Future = executor.submit(long_task, message)
    background_tasks.add_task(lambda: future.result())
    return {"task_id": id(future)}

def main():
    import uvicorn
    uvicorn.run(app, host=args.listen, port=args.port)

if __name__ == "__main__":
    main()