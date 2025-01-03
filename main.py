import os
from comfy_api.cli_args import args
from comfy_api.logger import setup_logger

if __name__ == "__main__":
    os.environ["PYTHONIOENCODING"] = "utf-8"

setup_logger(log_level="INFO", use_stdout=True)

def main():
    from app.server import app
    import uvicorn
    uvicorn.run(app, host=args.listen, port=args.port)

if __name__ == "__main__":
    main()