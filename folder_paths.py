
import os
import logging

supported_pt_extensions: set[str] = {'.ckpt', '.pt', '.bin', '.pth', '.safetensors', '.pkl', '.sft'}

folder_names_and_paths: dict[str, tuple[list[str], set[str]]] = {}

base_path = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(base_path, "models")

folder_names_and_paths["checkpoints"] = ([os.path.join(models_dir, "checkpoints")], supported_pt_extensions)
folder_names_and_paths["configs"] = ([os.path.join(models_dir, "configs")], [".yaml"])

folder_names_and_paths["loras"] = ([os.path.join(models_dir, "loras")], supported_pt_extensions)
folder_names_and_paths["vae"] = ([os.path.join(models_dir, "vae")], supported_pt_extensions)
folder_names_and_paths["text_encoders"] = ([os.path.join(models_dir, "text_encoders"), os.path.join(models_dir, "clip")], supported_pt_extensions)
folder_names_and_paths["diffusion_models"] = ([os.path.join(models_dir, "unet"), os.path.join(models_dir, "diffusion_models")], supported_pt_extensions)
folder_names_and_paths["clip_vision"] = ([os.path.join(models_dir, "clip_vision")], supported_pt_extensions)
folder_names_and_paths["style_models"] = ([os.path.join(models_dir, "style_models")], supported_pt_extensions)
folder_names_and_paths["embeddings"] = ([os.path.join(models_dir, "embeddings")], supported_pt_extensions)
folder_names_and_paths["diffusers"] = ([os.path.join(models_dir, "diffusers")], ["folder"])
folder_names_and_paths["vae_approx"] = ([os.path.join(models_dir, "vae_approx")], supported_pt_extensions)

folder_names_and_paths["controlnet"] = ([os.path.join(models_dir, "controlnet"), os.path.join(models_dir, "t2i_adapter")], supported_pt_extensions)
folder_names_and_paths["gligen"] = ([os.path.join(models_dir, "gligen")], supported_pt_extensions)

folder_names_and_paths["upscale_models"] = ([os.path.join(models_dir, "upscale_models")], supported_pt_extensions)

folder_names_and_paths["custom_nodes"] = ([os.path.join(base_path, "custom_nodes")], set())

folder_names_and_paths["hypernetworks"] = ([os.path.join(models_dir, "hypernetworks")], supported_pt_extensions)

folder_names_and_paths["photomaker"] = ([os.path.join(models_dir, "photomaker")], supported_pt_extensions)

folder_names_and_paths["classifiers"] = ([os.path.join(models_dir, "classifiers")], {""})

output_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "outputs")
temp_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp")
input_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "inputs")
user_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "user")

uploads_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "uploads")

def map_legacy(folder_name: str) -> str:
    legacy = {"unet": "diffusion_models",
              "clip": "text_encoders"}
    return legacy.get(folder_name, folder_name)

def get_full_path(folder_name: str, filename: str) -> str | None:
    global folder_names_and_paths
    folder_name = map_legacy(folder_name)
    if folder_name not in folder_names_and_paths:
        return None
    folders = folder_names_and_paths[folder_name]
    filename = os.path.relpath(os.path.join("/", filename), "/")
    for x in folders[0]:
        full_path = os.path.join(x, filename)
        if os.path.isfile(full_path):
            return full_path
        elif os.path.islink(full_path):
            logging.warning("WARNING path {} exists but doesn't link anywhere, skipping.".format(full_path))

    return None