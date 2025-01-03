from nodes.basic_nodes import Nodes
from comfy_api.imagefunc import *
from utils import random_seed
import folder_paths
import os


class LipstickColorTringFlow:
    def __init__(self) -> None:
        self.nodes = Nodes()
        
    def run(self, repaint_image_path, reference_image_path, unet_model, vae_model, clip_model, style_model, clip_vision_model, dino_model, sam_model):
        print("run start...")
        repaint_image_ts, reference_image_ts = self.__load_images(repaint_image_path, reference_image_path)
        with torch.inference_mode():
            print("segment start...")
            seg_mask_ts, _, _ = self.__clip_auto_segment(repaint_image_ts, dino_model, sam_model)

        seg_mask_ts = seg_mask_ts.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3)

        # print(type(seg_mask_ts))
        # print(type(repaint_image_ts))
        # print(type(reference_image_ts))

        print("image_normalization start...")
        repaint_image_ts = self.__image_normalization(repaint_image_ts, 1024, 1024)
        seg_mask_ts = self.__image_normalization(seg_mask_ts, 1024, 1024)
        reference_image_ts = self.__image_normalization(reference_image_ts, 1024, 1024)

        # print('seg_mask_ts size:', self.nodes.get_image_size(seg_mask_ts))
        # print('repaint_image size:', self.nodes.get_image_size(repaint_image_ts))
        # print('reference_image size:', self.nodes.get_image_size(reference_image_ts))

        # print("seg_mask_ts.shape:", seg_mask_ts.shape)
        # print("reference_image_ts.shape:", reference_image_ts.shape)

        # tensor2pil(repaint_image_ts).show()

        concanate_image, = self.nodes.image_concanate(reference_image_ts, repaint_image_ts, 'right', True)
        # tensor2pil(concanate_image).show()

        w, h, _ = self.nodes.get_image_size(reference_image_ts)
        reference_empty_mask_image, = self.nodes.generate_empty_image(w, h, 1, 0)
        # print(type(reference_empty_mask_ts))

        # print("reference_empty_mask_image.shape:", reference_empty_mask_image.shape)
        # print("seg_mask_ts.shape:", seg_mask_ts.shape)

        concanate_mask, = self.nodes.image_concanate(reference_empty_mask_image, seg_mask_ts, 'right', True)
        # tensor2pil(concanate_mask).show()

        print("inference start...")
        with torch.inference_mode():
            prompt_text = ''
            conditioning, = self.nodes.clip_text_encode(clip_model, prompt_text)
            conditioning_negative, = self.nodes.conditioning_zero_out(conditioning)
            print("redux start...")
            conditioning = self.__redux_vision(conditioning, reference_image_ts, style_model, clip_vision_model)
            conditioning_positive, = self.nodes.flux_guidance(conditioning, 30)

            # print("concanate_image.shape:", concanate_image.shape)
            # print("concanate_mask.shape:", concanate_mask.shape)

            concanate_mask = image2mask(tensor2pil(concanate_mask))

            conditioning_positive, conditioning_negative, latent = self.nodes.inpaint_model_contitioning(
                positive=conditioning_positive, 
                negative=conditioning_negative, 
                pixels=concanate_image, 
                vae=vae_model, 
                mask=concanate_mask, 
                noise_mask=True,
            )

            seed = random_seed(0, 99999999999999999)
            print(f"seed: {seed}")

            print("k_sampler start...")
            out_latent, = self.nodes.k_sampler(
                model=unet_model, 
                seed=seed, 
                steps=20, 
                cfg=1.0, 
                sampler_name='euler', 
                scheduler='normal', 
                positive=conditioning_positive, 
                negative=conditioning_negative, 
                latent_image=latent,
                denoise=1.0,
            )

            out_image, = self.nodes.vae_decode(vae_model, out_latent)

            save_path = os.path.join(folder_paths.output_directory, 'inpaint_'+str(seed)+'.png')
            tensor_to_pil_image(out_image).save(save_path)

    def __redux_vision(self, conditioning, reference_image_ts, style_model, clip_vision_model):
        clip_vision_output, = self.nodes.clip_vision_encode(clip_vision_model, reference_image_ts, 'center')
        conditioning, = self.nodes.apply_stylemodel(conditioning, style_model, clip_vision_output, 1.1, 'multiply')
        return conditioning

    def __load_images(self, repaint_image_path, reference_image_path) -> None:
        reference_image, _ = self.nodes.load_image(reference_image_path)
        repaint_image, _ = self.nodes.load_image(repaint_image_path)
        return (repaint_image, reference_image)

    def __clip_auto_segment(self, repaint_image, dino_model, sam_model):
        prompt = "lip"
        image_tensor, mask_tensor = self.nodes.do_sam(dino_model, sam_model, repaint_image, prompt, 0.25)
        growed_mask, = self.__grow_mask(mask_tensor)
        return (growed_mask, image_tensor, mask_tensor)

    def __image_normalization(self, image, target_width=512, target_height=512, x_offset=0, y_offset=0, mode='no'):
        output_images, = self.nodes.constrain_image(image, target_width, target_height, x_offset, y_offset, mode)
        if not output_images or len(output_images) == 0:
            raise ValueError("constrain_image returned an empty list")
        return output_images[0]

    def __grow_mask(self, mask, expand_piexls=20):
        exp_mask, = expand_mask(mask, expand_piexls, True)
        return exp_mask
    