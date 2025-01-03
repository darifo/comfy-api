from app.workflows.lipstick_color_tring import LipstickColorTringFlow
from fastapi import FastAPI, BackgroundTasks
from concurrent.futures import ThreadPoolExecutor, Future


app = FastAPI()
executor = ThreadPoolExecutor(max_workers=5)

from models_loader import models
dino_model, sam, unet, clip, vae, style_model, clip_vision_model = models.load_models()


def lipstick(message):
    LipstickColorTringFlow().run(
        "./inputs/person.jpg", 
        "./inputs/kouhong_cankao.jpg", 
        unet, 
        vae, 
        clip, 
        style_model, 
        clip_vision_model, 
        dino_model, 
        sam,
    )
    return f"Hello, {message}! finished"


@app.post("/prompt_queue/lipstick")
async def lipstick_task(message: str):
    future: Future = executor.submit(lipstick, message)
    result = future.result()
    return {"result": result}