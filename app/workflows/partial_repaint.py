from nodes.basic_nodes import Nodes
from comfy_api.imagefunc import *
from utils import random_seed
import folder_paths
import os
import time

class PartialRepaintWorkflow:
    def __init__(self) -> None:
        self.nodes = Nodes()
        self._flux_guidance = 30
        self._steps = 10
        self._cfg = 1.0
        self._denoise = 1.0
        self._sampler_name='euler' 
        self._scheduler='normal'
        self._redux_strength = 1.2
        self._sam_threshold = 0.25
        self._prompt_text = ''

        self.dino_model, = self.nodes.groundingdino_model_loader("GroundingDINO_SwinT_OGC (694MB)")
        self.sam, = self.nodes.sam_model_loader("sam_vit_b (375MB)")
        self.unet, = self.nodes.load_unet("flux1-fill-dev.safetensors", "fp8_e4m3fn")
        self.clip, = self.nodes.load_dual_clip("clip_l.safetensors", "t5xxl_fp8_e4m3fn.safetensors", "flux")
        self.vae, = self.nodes.load_vae("ae.safetensors")
        self.style_model, = self.nodes.load_stylemodel("flux1-redux-dev.safetensors")
        self.clip_vision_model, = self.nodes.clip_vision_model_load("sigclip_vision_patch14_384.safetensors")
        
        
    def run(self, repaint_image_path, reference_image_path, prompt_text):
        print("load images...")
        
        t1 = time.time()
        repaint_image_ts, reference_image_ts = self.__load_images(repaint_image_path, reference_image_path)
        t2 = time.time()
        print(f"load images time: {t2-t1}")

        with torch.inference_mode():
            print("segment repaint_image...")
            t3 = time.time()
            # reference_seg_mask_ts, _, _ = self.__clip_auto_segment(reference_image_ts, self.dino_model, self.sam, prompt=prompt_text)
            repaint_seg_mask_ts, _, _ = self.__clip_auto_segment(repaint_image_ts, self.dino_model, self.sam, prompt=prompt_text)
        t4 = time.time()
        print(f"segment repaint_image time: {t4-t3}")
        # 升维多通道以便尺寸归一和拼接
        repaint_seg_mask_ts = repaint_seg_mask_ts.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3)
        # reference_seg_mask_ts = reference_seg_mask_ts.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3)

        print(f"repaint_seg_mask_ts.shape: {repaint_seg_mask_ts.shape}")
        # print(f"reference_seg_mask_ts.shape: {reference_seg_mask_ts.shape}")

        # _, cropped_out, _, _, _ = self.nodes.image_crop_by_mask(
        #     reference_seg_mask_ts,
        #     reference_image_ts,
        #     crop_size_mult=1.000,
        #     bbox_smooth_alpha=0.90
        # )

        self._unified_width, self._unified_hight, _ = self.nodes.get_image_size(repaint_image_ts)

        print("image_normalization start...")
        t5 = time.time()
        repaint_image_ts = self.__image_normalization(repaint_image_ts)
        repaint_seg_mask_ts = self.__image_normalization(repaint_seg_mask_ts)
        reference_image_ts = self.__image_normalization(reference_image_ts)

        # repaint_image_ts, = self.nodes.image_scale_by_wh(repaint_image_ts, 'lanczos', 768, 1024, False)
        # repaint_seg_mask_ts, = self.nodes.image_scale_by_wh(repaint_seg_mask_ts, 'lanczos', 768, 1024, False)
        # reference_image_ts, = self.nodes.image_scale_by_wh(reference_image_ts, 'lanczos', 768, 1024, False)

        # output_images, = self.nodes.constrain_image(repaint_image_ts, 1024, 1024, 512, 1024, 'no')
        # repaint_image_ts = output_images[0]
        # output_images, = self.nodes.constrain_image(repaint_seg_mask_ts, 1024, 1024, 512, 1024, 'no')
        # repaint_seg_mask_ts = output_images[0]
        # output_images, = self.nodes.constrain_image(reference_image_ts, 1024, 1024, 512, 1024, 'no')
        # reference_image_ts = output_images[0]

        t6 = time.time()

        # 打印size
        print("repaint_image_ts:", repaint_image_ts.shape)
        print("repaint_seg_mask_ts:", repaint_seg_mask_ts.shape)
        print("reference_image_ts:", reference_image_ts.shape)

        concanate_image, = self.nodes.image_concanate(reference_image_ts, repaint_image_ts, 'right', True)
        t7 = time.time()
        # concanate_image, = self.nodes.image_scale_by_wh(concanate_image, 'lanczos', 768*2, 1024, False)
        # output_images, = self.nodes.constrain_image(concanate_image, 1024, 2048, 0, 0, 'no')
        # concanate_image, = self.nodes.image_upscale_by_coefficient(concanate_image, 'lanczos', 0.5)
        # concanate_image = output_images[0]
        print("concanate_image:", concanate_image.shape)
        # tensor2pil(concanate_image).show()

        repaint_image_normal_w, repaint_image_normal_h, _ = self.nodes.get_image_size(repaint_image_ts)

        reference_image_normal_w, reference_image_normal_h, _ = self.nodes.get_image_size(reference_image_ts)
        reference_empty_mask_image, = self.nodes.generate_empty_image(reference_image_normal_w, reference_image_normal_h, 1, 0)
        # print(type(reference_empty_mask_ts))

        concanate_mask, = self.nodes.image_concanate(reference_empty_mask_image, repaint_seg_mask_ts, 'right', True)
        # concanate_mask, = self.nodes.image_scale_by_wh(concanate_mask, 'lanczos', 768*2, 1024, False)
        # output_mask_images, = self.nodes.constrain_image(concanate_mask, 1024, 2048, 0, 0, 'no')
        # concanate_mask, = self.nodes.image_upscale_by_coefficient(concanate_mask, 'lanczos', 0.5)
        # concanate_mask = output_mask_images[0]
        print("concanate_mask:", concanate_mask.shape)
        # tensor2pil(concanate_mask).show()

        print("inference start...")
        t8 = time.time()
        with torch.inference_mode():
            conditioning, = self.nodes.clip_text_encode(self.clip, self._prompt_text)
            conditioning_negative, = self.nodes.conditioning_zero_out(conditioning)
            t9 = time.time()
            print(f"clip_text_encode time: {t9-t8}")

            print("redux start...")
            conditioning = self.__redux_vision(conditioning, reference_image_ts, self.style_model, self.clip_vision_model)
            conditioning_positive, = self.nodes.flux_guidance(conditioning, self._flux_guidance)
            t10 = time.time()
            print(f"redux time: {t10-t9}")

            concanate_mask = image2mask(tensor2pil(concanate_mask))
            conditioning_positive, conditioning_negative, latent = self.nodes.inpaint_model_contitioning(
                positive=conditioning_positive, 
                negative=conditioning_negative, 
                pixels=concanate_image, 
                vae=self.vae, 
                mask=concanate_mask, 
                noise_mask=True,
            )
            t11 = time.time()
            print(f"inpaint_model_contitioning time: {t11-t10}")

            seed = random_seed(0, 99999999999999999)
            print(f"seed: {seed}")

            print("k_sampler start...")
            out_latent, = self.nodes.k_sampler(
                model=self.unet, 
                seed=seed, 
                steps=self._steps, 
                cfg=self._cfg, 
                sampler_name=self._sampler_name, 
                scheduler=self._scheduler, 
                positive=conditioning_positive, 
                negative=conditioning_negative, 
                latent_image=latent,
                denoise=self._denoise,
            )
            t12 = time.time()
            print(f"k_sampler time: {t12-t11}")

            out_image, = self.nodes.vae_decode(self.vae, out_latent)
            t13 = time.time()
            print(f"vae_decode time: {t13-t12}")

            sid = str(seed)
            concanate_image_save_name = 'inpaint_concanate_'+sid+'.png'
            self.__save_image(out_image, concanate_image_save_name)

            # 裁剪
            out_image, = self.nodes.crop(out_image, repaint_image_normal_w, repaint_image_normal_h, reference_image_normal_w, 0)

            # 恢复到最初的尺寸
            out_image, = self.nodes.image_scale_by_wh(out_image, 'lanczos', self._unified_width, self._unified_hight, False)

            inpaint_image_save_name = 'inpaint_'+sid+'.png'
            self.__save_image(out_image, inpaint_image_save_name)

            return sid
        
    def __save_image(self, image, image_name):
        save_path = os.path.join(folder_paths.output_directory, image_name)
        output_pil_image = tensor_to_pil_image(image)
        output_pil_image.save(save_path)
        print(f"save image: {save_path}")
        return save_path

    def __redux_vision(self, conditioning, reference_image_ts, style_model, clip_vision_model):
        clip_vision_output, = self.nodes.clip_vision_encode(clip_vision_model, reference_image_ts, 'center')
        conditioning, = self.nodes.apply_stylemodel(conditioning, style_model, clip_vision_output, self._redux_strength, 'multiply')
        return conditioning

    def __load_images(self, repaint_image_path, reference_image_path) -> None:
        reference_image, _ = self.nodes.load_image(reference_image_path)
        repaint_image, _ = self.nodes.load_image(repaint_image_path)
        return (repaint_image, reference_image)

    def __clip_auto_segment(self, repaint_image, dino_model, sam_model, prompt):
        image_tensor, mask_tensor = self.nodes.do_sam(dino_model, sam_model, repaint_image, prompt, self._sam_threshold)
        growed_mask, = self.__grow_mask(mask_tensor)
        return (growed_mask, image_tensor, mask_tensor)

    def __image_normalization(self, image):
        # 缩放 高度要保持一致 以便横向拼接 1024在flux模型上效果最好
        image, = self.nodes.image_scale_by_wh(image, 'lanczos', 0, self._unified_hight, False)
        output_images, = self.nodes.constrain_image(image, 512, 1024, 512, 1024, 'no')
        if not output_images or len(output_images) == 0:
            raise ValueError("constrain_image returned an empty list")
        return output_images[0]
    
    def __grow_mask(self, mask, expand_piexls=20):
        exp_mask, = expand_mask(mask, expand_piexls, True)
        return exp_mask
    