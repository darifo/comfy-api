from nodes.basic_nodes import Nodes

class ModelLoader:
    def __init__(self):
        self.nodes = Nodes()

    def load_models(self):
        dino_model, = self.nodes.groundingdino_model_loader("GroundingDINO_SwinT_OGC (694MB)")
        sam, = self.nodes.sam_model_loader("sam_vit_b (375MB)")
        unet, = self.nodes.load_unet("flux1-fill-dev.safetensors", "fp8_e4m3fn")
        clip, = self.nodes.load_dual_clip("clip_l.safetensors", "t5xxl_fp8_e4m3fn.safetensors", "flux")
        vae, = self.nodes.load_vae("ae.safetensors")
        style_model, = self.nodes.load_stylemodel("flux1-redux-dev.safetensors")
        clip_vision_model, = self.nodes.clip_vision_model_load("sigclip_vision_patch14_384.safetensors")
        print("All models loaded successfully.")
        return dino_model, sam, unet, clip, vae, style_model, clip_vision_model

models = ModelLoader()