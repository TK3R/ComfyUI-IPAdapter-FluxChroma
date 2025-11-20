import torch
import os
import logging
import folder_paths
from transformers import AutoProcessor, SiglipVisionModel
from PIL import Image
import numpy as np
from .attention_processor import IPAFluxAttnProcessor2_0
from .utils import is_model_patched, FluxUpdateModules

MODELS_DIR = os.path.join(folder_paths.models_dir, "ipadapter-flux")
CLIP_VISION_DIR = os.path.join(folder_paths.models_dir, "clip_vision")
if "ipadapter-flux" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter-flux"]
    MODELS_DIR = current_paths[0]
    
folder_paths.folder_names_and_paths["ipadapter-flux"] = (current_paths, folder_paths.supported_pt_extensions)

class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class InstantXFluxIPAdapterModel:
    def __init__(self, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        # load image encoder
        try:
            self.image_encoder = SiglipVisionModel.from_pretrained(CLIP_VISION_DIR+"/"+self.image_encoder_path.split("/")[-1]).to(self.device, dtype=torch.float16)
            self.clip_image_processor = AutoProcessor.from_pretrained(CLIP_VISION_DIR+"/"+self.image_encoder_path.split("/")[-1])
        except:
            try:
                self.image_encoder = SiglipVisionModel.from_pretrained(self.image_encoder_path).to(self.device, dtype=torch.float16)
                self.clip_image_processor = AutoProcessor.from_pretrained(self.image_encoder_path)
            except:
                raise Exception(f"Failed to load clip image processor for {self.image_encoder_path}, please check the huggingface cache directory or clip_vision directory")
        # state_dict
        state_dict = torch.load(os.path.join(MODELS_DIR,self.ip_ckpt), map_location="cpu")
        self.joint_attention_dim = 4096
        self.hidden_size = 3072
        # init projection model
        self.image_proj_model = MLPProjModel(
            cross_attention_dim=self.joint_attention_dim, # 4096
            id_embeddings_dim=1152, 
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        # init ipadapter model
        self.ip_attn_procs = self.init_ip_adapter()
        ip_layers = torch.nn.ModuleList(self.ip_attn_procs.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=True)
        del state_dict

    def init_ip_adapter(self, weight=1.0, timestep_percent_range=(0.0, 1.0)):
        ip_attn_procs = {} # 19+38=57
        dsb_count = 19 #len(flux_model.diffusion_model.double_blocks)
        for i in range(dsb_count):
            name = f"double_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=self.hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale = weight,
                    timestep_range = timestep_percent_range
                ).to(self.device, dtype=torch.float16)
        ssb_count = 38 #len(flux_model.diffusion_model.single_blocks)
        for i in range(ssb_count):
            name = f"single_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
                    hidden_size=self.hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale = weight,
                    timestep_range = timestep_percent_range
                ).to(self.device, dtype=torch.float16)
        return ip_attn_procs
    
    def update_ip_adapter(self, flux_model, weight, timestep_percent_range=(0.0, 1.0)):
        s = flux_model.model_sampling
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        timestep_range = (percent_to_timestep_function(timestep_percent_range[0]), percent_to_timestep_function(timestep_percent_range[1]))
        for ip_layer in self.ip_attn_procs.values():
            ip_layer.scale = weight
            ip_layer.timestep_range = timestep_range

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            # Ensure encoder is on the correct device (it may have been offloaded to CPU)
            if self.image_encoder.device != self.device:
                self.image_encoder.to(self.device)
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.image_encoder.dtype)).pooler_output
            clip_image_embeds = clip_image_embeds.to(dtype=torch.float16)
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        # Ensure projection model is on the correct device
        proj_device = next(self.image_proj_model.parameters()).device
        if proj_device != self.device:
            self.image_proj_model.to(self.device)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds

class IPAdapterFluxChromaLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "ipadapter": (folder_paths.get_filename_list("ipadapter-flux"),),
                "clip_vision": (["google/siglip-so400m-patch14-384"],),
                "provider": (["cuda", "cpu", "mps"],),
            }
        }
    RETURN_TYPES = ("IP_ADAPTER_FLUX_INSTANTX",)
    RETURN_NAMES = ("ipadapterFlux",)
    FUNCTION = "load_model"
    CATEGORY = "InstantXNodes"

    def load_model(self, ipadapter, clip_vision, provider):
        logging.info("Loading InstantX IPAdapter Flux model.")
        model = InstantXFluxIPAdapterModel(image_encoder_path=clip_vision, ip_ckpt=ipadapter, device=provider, num_tokens=128)
        return (model,)

class ApplyIPAdapterFluxChroma:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter_flux": ("IP_ADAPTER_FLUX_INSTANTX", ),
                "image": ("IMAGE", ),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "offload_preprocessing": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"})
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter_flux"
    CATEGORY = "InstantXNodes"

    def apply_ipadapter_flux(self, model, ipadapter_flux, image, weight, start_percent, end_percent, offload_preprocessing=True):
        # convert image to pillow
        pil_image = image.numpy()[0] * 255.0
        pil_image = Image.fromarray(pil_image.astype(np.uint8))
        # initialize ipadapter
        ipadapter_flux.update_ip_adapter(model.model, weight, (start_percent, end_percent))
        # process control image 
        image_prompt_embeds = ipadapter_flux.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=None
        )
        # set model
        is_patched = is_model_patched(model.model)
        bi = model.clone()
        FluxUpdateModules(bi, ipadapter_flux.ip_attn_procs, image_prompt_embeds, is_patched)
        
        # Note: We don't register image_encoder and image_proj_model with ModelPatcher
        # because they have read-only device properties and aren't compatible with ComfyUI's
        # automatic memory management. We handle them manually via offloading below.
        # Only the attention processors are registered since they're standard torch modules.
        
        # Register attention processors with memory management
        try:
            from comfy.model_patcher import ModelPatcher
            additional_models = []
            
            # Only wrap attention processors (57 modules: 19 double + 38 single blocks)
            # These are standard PyTorch modules that work with ModelPatcher
            if hasattr(ipadapter_flux, 'ip_attn_procs') and ipadapter_flux.ip_attn_procs:
                # Get device from first attention processor
                first_attn_proc = next(iter(ipadapter_flux.ip_attn_procs.values()))
                attn_device = next(first_attn_proc.parameters()).device
                # Wrap all attention processors in a ModuleList for tracking
                ip_layers = torch.nn.ModuleList(ipadapter_flux.ip_attn_procs.values())
                attn_patcher = ModelPatcher(ip_layers, attn_device, torch.device("cpu"))
                additional_models.append(attn_patcher)
            
            # Register with the model
            if additional_models:
                bi.set_additional_models("ipadapter_flux", additional_models)
        except Exception as e:
            import traceback
            logging.warning(f"IPAdapter-FluxChroma: Could not register attention processors: {e}")
            logging.debug(traceback.format_exc())
        
        # Manually offload preprocessing models to CPU now that embeddings are generated
        # These are only needed during image encoding, not during inference
        # This saves ~1.2GB VRAM (image_encoder ~1.1GB + image_proj_model ~0.1GB)
        if offload_preprocessing:
            try:
                offloaded_count = 0
                if hasattr(ipadapter_flux.image_encoder, 'cpu'):
                    original_device = ipadapter_flux.image_encoder.device
                    ipadapter_flux.image_encoder.to('cpu')
                    offloaded_count += 1
                    logging.info(f"IPAdapter-FluxChroma: Offloaded SiglipVisionModel from {original_device} to CPU (~1.1GB VRAM freed)")
                if hasattr(ipadapter_flux.image_proj_model, 'cpu'):
                    original_device = next(ipadapter_flux.image_proj_model.parameters()).device
                    ipadapter_flux.image_proj_model.to('cpu')
                    offloaded_count += 1
                    logging.info(f"IPAdapter-FluxChroma: Offloaded projection model from {original_device} to CPU (~0.1GB VRAM freed)")
                # Clear CUDA cache to immediately reclaim memory
                if offloaded_count > 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                logging.warning(f"IPAdapter-FluxChroma: Could not offload preprocessing models: {e}")
                logging.debug(traceback.format_exc())
        else:
            # Move models back to GPU if they were previously offloaded
            try:
                moved_count = 0
                if hasattr(ipadapter_flux.image_encoder, 'to'):
                    current_device = ipadapter_flux.image_encoder.device
                    if current_device.type == 'cpu':
                        target_device = model.load_device
                        ipadapter_flux.image_encoder.to(target_device)
                        moved_count += 1
                        logging.info(f"IPAdapter-FluxChroma: Moved SiglipVisionModel from CPU back to {target_device}")
                if hasattr(ipadapter_flux.image_proj_model, 'to'):
                    current_device = next(ipadapter_flux.image_proj_model.parameters()).device
                    if current_device.type == 'cpu':
                        target_device = model.load_device
                        ipadapter_flux.image_proj_model.to(target_device)
                        moved_count += 1
                        logging.info(f"IPAdapter-FluxChroma: Moved projection model from CPU back to {target_device}")
            except Exception as e:
                import traceback
                logging.warning(f"IPAdapter-FluxChroma: Could not move preprocessing models back to GPU: {e}")
                logging.debug(traceback.format_exc())
        
        return (bi,)

NODE_CLASS_MAPPINGS = {
    "IPAdapterFluxChromaLoader": IPAdapterFluxChromaLoader,
    "ApplyIPAdapterFluxChroma": ApplyIPAdapterFluxChroma,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterFluxChromaLoader": "Load IPAdapter Flux/Chroma Model",
    "ApplyIPAdapterFluxChroma": "Apply IPAdapter Flux/Chroma Model",
}
