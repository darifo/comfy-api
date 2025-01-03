import torch
import os
import comfy_api.sd
import comfy_api.comfy_utils
import comfy_api.sample
import comfy_api.clip_vision
from comfy_api.comfy_utils import common_upscale
from . import node_helpers
import numpy as np
from comfy_api.thirds_nodes.segment_anything import load_sam_model, load_groundingdino_model, groundingdino_predict, sam_segment
from PIL import Image, ImageOps, ImageSequence
import folder_paths
from comfy_api.imagefunc import *

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy_api.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy_api.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = None
    disable_pbar = not comfy_api.comfy_utils.PROGRESS_BAR_ENABLED
    samples = comfy_api.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )


class Nodes:
    def __init__(self):
        self.model_base_path = folder_paths.models_dir
    # 加载unet
    def load_unet(self, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = os.path.join(self.model_base_path, "unet", unet_name)
        model = comfy_api.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)
    # 加载双clip
    def load_dual_clip(self, clip_name1, clip_name2, type):
            clip_path1 = os.path.join(self.model_base_path, "text_encoder", clip_name1)
            clip_path2 = os.path.join(self.model_base_path, "text_encoder", clip_name2)
            if type == "sdxl":
                clip_type = comfy_api.sd.CLIPType.STABLE_DIFFUSION
            elif type == "sd3":
                clip_type = comfy_api.sd.CLIPType.SD3
            elif type == "flux":
                clip_type = comfy_api.sd.CLIPType.FLUX
            elif type == "hunyuan_video":
                clip_type = comfy_api.sd.CLIPType.HUNYUAN_VIDEO

            clip = comfy_api.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=None, clip_type=clip_type)
            return (clip,)
    # 加载vae
    def load_vae(self, vae_name):
        vae_path = os.path.join(self.model_base_path, "vae", vae_name)
        sd = comfy_api.comfy_utils.load_torch_file(vae_path)
        vae = comfy_api.sd.VAE(sd=sd)
        return (vae,)

    # CLIP文本编码器
    def clip_text_encode(self, clip, text):
        tokens = clip.tokenize(text)
        return (clip.encode_from_tokens_scheduled(tokens), )

    # CLIP视觉模型加载器 
    def clip_vision_model_load(self, clip_name):
        clip_path = os.path.join(self.model_base_path, "clip_vision", clip_name)
        clip_vision = comfy_api.clip_vision.load(clip_path)
        return (clip_vision,)

    # CLIP视觉编码
    def clip_vision_encode(self, clip_vision, image, crop):
        crop_image = True
        if crop != "center":
            crop_image = False
        output = clip_vision.encode_image(image, crop=crop_image)
        return (output,)

    # 加载风格模型
    def load_stylemodel(self, style_model_name):
        style_model_path = os.path.join(self.model_base_path, "style_models", style_model_name)
        style_model = comfy_api.sd.load_style_model(style_model_path)
        return (style_model,)

    # 应用风格模型 
    def apply_stylemodel(self, conditioning, style_model, clip_vision_output, strength, strength_type):
            cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
            if strength_type == "multiply":
                cond *= strength

            n = cond.shape[1]
            c_out = []
            for t in conditioning:
                (txt, keys) = t
                keys = keys.copy()
                if strength_type == "attn_bias" and strength != 1.0:
                    # math.log raises an error if the argument is zero
                    # torch.log returns -inf, which is what we want
                    attn_bias = torch.log(torch.Tensor([strength]))
                    # get the size of the mask image
                    mask_ref_size = keys.get("attention_mask_img_shape", (1, 1))
                    n_ref = mask_ref_size[0] * mask_ref_size[1]
                    n_txt = txt.shape[1]
                    # grab the existing mask
                    mask = keys.get("attention_mask", None)
                    # create a default mask if it doesn't exist
                    if mask is None:
                        mask = torch.zeros((txt.shape[0], n_txt + n_ref, n_txt + n_ref), dtype=torch.float16)
                    # convert the mask dtype, because it might be boolean
                    # we want it to be interpreted as a bias
                    if mask.dtype == torch.bool:
                        # log(True) = log(1) = 0
                        # log(False) = log(0) = -inf
                        mask = torch.log(mask.to(dtype=torch.float16))
                    # now we make the mask bigger to add space for our new tokens
                    new_mask = torch.zeros((txt.shape[0], n_txt + n + n_ref, n_txt + n + n_ref), dtype=torch.float16)
                    # copy over the old mask, in quandrants
                    new_mask[:, :n_txt, :n_txt] = mask[:, :n_txt, :n_txt]
                    new_mask[:, :n_txt, n_txt+n:] = mask[:, :n_txt, n_txt:]
                    new_mask[:, n_txt+n:, :n_txt] = mask[:, n_txt:, :n_txt]
                    new_mask[:, n_txt+n:, n_txt+n:] = mask[:, n_txt:, n_txt:]
                    # now fill in the attention bias to our redux tokens
                    new_mask[:, :n_txt, n_txt:n_txt+n] = attn_bias
                    new_mask[:, n_txt+n:, n_txt:n_txt+n] = attn_bias
                    keys["attention_mask"] = new_mask.to(txt.device)
                    keys["attention_mask_img_shape"] = mask_ref_size

                c_out.append([torch.cat((txt, cond), dim=1), keys])

            return (c_out,)

    # FLUX引导
    def flux_guidance(self, conditioning, guidance):
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return (c, )
    
    # 条件零化
    def conditioning_zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            n = [torch.zeros_like(t[0]), d]
            c.append(n)
        return (c, )
    
    # 内补条件模型
    def inpaint_model_contitioning(self, positive, negative, pixels, vae, mask, noise_mask=True):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        orig_pixels = pixels
        pixels = orig_pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m
            pixels[:,:,:,i] += 0.5
        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)

        out_latent = {}

        out_latent["samples"] = orig_latent
        if noise_mask:
            out_latent["noise_mask"] = mask

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent,
                                                                    "concat_mask": mask})
            out.append(c)
        return (out[0], out[1], out_latent)
    
    # K采样器
    def k_sampler(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
    
    # VAE解码
    def vae_decode(self, vae, samples):
        images = vae.decode(samples["samples"])
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images, )
    
    # 图片缩放按宽高裁剪
    def image_scale_by_wh(self, image, upscale_method, width, height, crop):
        if width == 0 and height == 0:
            s = image
        else:
            samples = image.movedim(-1,1)

            if width == 0:
                width = max(1, round(samples.shape[3] * height / samples.shape[2]))
            elif height == 0:
                height = max(1, round(samples.shape[2] * width / samples.shape[3]))

            s = comfy_api.comfy_utils.common_upscale(samples, width, height, upscale_method, crop)
            s = s.movedim(1,-1)
        return (s,)
    
    # 限制图像的宽高
    def constrain_image(self, images, max_width, max_height, min_width, min_height, crop_if_required):
            crop_if_required = crop_if_required == "yes"
            results = []
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")

                current_width, current_height = img.size
                aspect_ratio = current_width / current_height

                constrained_width = max(min(current_width, min_width), max_width)
                constrained_height = max(min(current_height, min_height), max_height)

                if constrained_width / constrained_height > aspect_ratio:
                    constrained_width = max(int(constrained_height * aspect_ratio), min_width)
                    if crop_if_required:
                        constrained_height = int(current_height / (current_width / constrained_width))
                else:
                    constrained_height = max(int(constrained_width / aspect_ratio), min_height)
                    if crop_if_required:
                        constrained_width = int(current_width / (current_height / constrained_height))

                resized_image = img.resize((constrained_width, constrained_height), Image.LANCZOS)

                if crop_if_required and (constrained_width > max_width or constrained_height > max_height):
                    left = max((constrained_width - max_width) // 2, 0)
                    top = max((constrained_height - max_height) // 2, 0)
                    right = min(constrained_width, max_width) + left
                    bottom = min(constrained_height, max_height) + top
                    resized_image = resized_image.crop((left, top, right, bottom))

                resized_image = np.array(resized_image).astype(np.float32) / 255.0
                resized_image = torch.from_numpy(resized_image)[None,]
                results.append(resized_image)
                    
            return (results,)
    
    # 图像合并
    def image_concanate(self, image1, image2, direction, match_image_size, first_image_shape=None):
        # Check if the batch sizes are different
        batch_size1 = image1.shape[0]
        batch_size2 = image2.shape[0]

        if batch_size1 != batch_size2:
            # Calculate the number of repetitions needed
            max_batch_size = max(batch_size1, batch_size2)
            repeats1 = max_batch_size // batch_size1
            repeats2 = max_batch_size // batch_size2
            
            # Repeat the images to match the largest batch size
            image1 = image1.repeat(repeats1, 1, 1, 1)
            image2 = image2.repeat(repeats2, 1, 1, 1)

        if match_image_size:
            # Use first_image_shape if provided; otherwise, default to image1's shape
            target_shape = first_image_shape if first_image_shape is not None else image1.shape

            original_height = image2.shape[1]
            original_width = image2.shape[2]
            original_aspect_ratio = original_width / original_height

            if direction in ['left', 'right']:
                # Match the height and adjust the width to preserve aspect ratio
                target_height = target_shape[1]  # B, H, W, C format
                target_width = int(target_height * original_aspect_ratio)
            elif direction in ['up', 'down']:
                # Match the width and adjust the height to preserve aspect ratio
                target_width = target_shape[2]  # B, H, W, C format
                target_height = int(target_width / original_aspect_ratio)
            
            # Adjust image2 to the expected format for common_upscale
            image2_for_upscale = image2.movedim(-1, 1)  # Move C to the second position (B, C, H, W)
            
            # Resize image2 to match the target size while preserving aspect ratio
            image2_resized = common_upscale(image2_for_upscale, target_width, target_height, "lanczos", "disabled")
            
            # Adjust image2 back to the original format (B, H, W, C) after resizing
            image2_resized = image2_resized.movedim(1, -1)
        else:
            image2_resized = image2

        # Ensure both images have the same number of channels
        channels_image1 = image1.shape[-1]
        channels_image2 = image2_resized.shape[-1]

        if channels_image1 != channels_image2:
            if channels_image1 < channels_image2:
                # Add alpha channel to image1 if image2 has it
                alpha_channel = torch.ones((*image1.shape[:-1], channels_image2 - channels_image1), device=image1.device)
                image1 = torch.cat((image1, alpha_channel), dim=-1)
            else:
                # Add alpha channel to image2 if image1 has it
                alpha_channel = torch.ones((*image2_resized.shape[:-1], channels_image1 - channels_image2), device=image2_resized.device)
                image2_resized = torch.cat((image2_resized, alpha_channel), dim=-1)


        # Concatenate based on the specified direction
        if direction == 'right':
            concatenated_image = torch.cat((image1, image2_resized), dim=2)  # Concatenate along width
        elif direction == 'down':
            concatenated_image = torch.cat((image1, image2_resized), dim=1)  # Concatenate along height
        elif direction == 'left':
            concatenated_image = torch.cat((image2_resized, image1), dim=2)  # Concatenate along width
        elif direction == 'up':
            concatenated_image = torch.cat((image2_resized, image1), dim=1)  # Concatenate along height
        return concatenated_image,

    # 获取图片尺寸
    def get_image_size(self, image,):
        if image.shape[0] > 0:
            image = torch.unsqueeze(image[0], 0)
        _image = tensor2pil(image)
        return (_image.width, _image.height, [_image.width, _image.height],)
    
    # 空图像
    def generate(self, width, height, batch_size=1, color=0):
        r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 0xFF)
        g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 0xFF)
        b = torch.full([batch_size, height, width, 1], ((color) & 0xFF) / 0xFF)
        return (torch.cat((r, g, b), dim=-1), )
    
    # 转换图像到mask
    def image_to_mask(self, image, channel):
        channels = ["red", "green", "blue", "alpha"]
        mask = image[:, :, :, channels.index(channel)]
        return (mask,)
    
    # 转换mask到图像
    def mask_to_image(self, mask):
        result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return (result,)

    # 图像裁切
    def crop(self, image, width, height, x, y):
        x = min(x, image.shape[2] - 1)
        y = min(y, image.shape[1] - 1)
        to_x = width + x
        to_y = height + y
        img = image[:,y:to_y, x:to_x, :]
        return (img,)
    
    # 加载图片
    def load_image(self, image_path):

        img = node_helpers.pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    # 分割模型加载
    def sam_model_loader(self, model_name):
        sam_model = load_sam_model(model_name)
        return (sam_model, )
    
    # GroundingDINO模型加载器
    def groundingdino_model_loader(self, model_name):
        dino_model = load_groundingdino_model(model_name)
        return (dino_model, )
    
    # SAM推理
    def do_sam(self, grounding_dino_model, sam_model, image, prompt, threshold):
        res_images = []
        res_masks = []
        for item in image:
            item = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            boxes = groundingdino_predict(
                grounding_dino_model,
                item,
                prompt,
                threshold
            )
            if boxes.shape[0] == 0:
                break
            (images, masks) = sam_segment(
                sam_model,
                item,
                boxes
            )
            res_images.extend(images)
            res_masks.extend(masks)
        if len(res_images) == 0:
            _, height, width, _ = image.size()
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return (empty_mask, empty_mask)
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))
    
    # 图像合成遮罩
    def image_composite_mask(self, destination, source, x, y, resize_source, mask = None):
        destination = destination.clone().movedim(-1, 1)
        output = node_helpers.composite(destination, source.movedim(-1, 1), x, y, mask, 1, resize_source).movedim(1, -1)
        return (output,)
    
    # 生成一张空图像
    def generate_empty_image(self, width, height, batch_size=1, color=0):
        r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 0xFF)
        g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 0xFF)
        b = torch.full([batch_size, height, width, 1], ((color) & 0xFF) / 0xFF)
        return (torch.cat((r, g, b), dim=-1), )