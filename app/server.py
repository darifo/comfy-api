from app.workflows.lipstick_color_tring import LipstickColorTringFlow
from fastapi import FastAPI, File, UploadFile
from concurrent.futures import ThreadPoolExecutor, Future
from models_loader import model_loader
import folder_paths
import os
import hashlib

# 加载模型
dino_model, sam, unet, clip, vae, style_model, clip_vision_model = model_loader.load_models()

def lipstick_trying_task(reference_image_path, repaint_image_path):
    base64_image, out_image_name = LipstickColorTringFlow().run(
            repaint_image_path, 
            reference_image_path, 
            unet, 
            vae, 
            clip, 
            style_model, 
            clip_vision_model, 
            dino_model, 
            sam,
        )
    return {"out_image_name": out_image_name}

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=1)

# 统一返回格式的方法
def response_format(data: dict, code: int = 0, message: str = 'success'):
    return {
        "code": code,
        "message": message,
        "data": data
    }

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    # 判断文件是否为空
    if not file.file:
        return response_format({}, 7001, "文件为空")
    if file.content_type not in ["image/png", "image/jpeg"]:
        return response_format({}, 7010, "文件格式错误")
    if len(await file.read()) > 1024 * 1024 * 2:
        return response_format({}, 7011, "文件大小超过2MB")

    try:
        # 计算文件 MD5 值
        file.file.seek(0)
        file_hash = hashlib.md5()
        while chunk := file.file.read(1024):
            file_hash.update(chunk)
        md5_value = file_hash.hexdigest()
        # 使用 MD5 值作为文件名
        new_filename = f"{md5_value}.{file.filename.split('.')[-1]}"
        # 将文件保存到指定路径
        save_path = os.path.join(folder_paths.uploads_directory, new_filename)
        with open(save_path, "wb") as buffer:
            file.file.seek(0)
            buffer.write(file.file.read())
    except Exception as e:
        return response_format({}, 7002, "文件上传失败")
    
    return response_format({"filename": new_filename})

@app.post("/workflows/lipstick")
async def lipstick_task(reference_image: str, repaint_image: str):
    # 校验参数
    if not reference_image:
        return response_format({}, 7003, "reference_image is required")
    if not repaint_image:
        return response_format({}, 7004, "repaint_image is required")

    reference_image_path = os.path.join(folder_paths.uploads_directory, reference_image)
    repaint_image_path = os.path.join(folder_paths.uploads_directory, repaint_image)

    if not os.path.exists(reference_image_path):
        return response_format({}, 7005, "reference_image not found")
    if not os.path.exists(repaint_image_path):
        return response_format({}, 7006, "repaint_image not found")
    
    try:
        future: Future = executor.submit(lipstick_trying_task, reference_image_path, repaint_image_path)
        result = future.result()
        return response_format(result)
    except Exception as e:
        return response_format({}, 7007, "task error")
