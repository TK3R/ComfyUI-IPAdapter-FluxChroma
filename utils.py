import torch
from torch import Tensor
from .flux.layers import DoubleStreamBlockIPA, SingleStreamBlockIPA
from comfy.ldm.flux.layers import timestep_embedding
from types import MethodType

def FluxUpdateModules(bi, ip_attn_procs, image_emb, is_patched):
    flux_model = bi.model
    
    # Check if this is ChromaRadiance - it's incompatible with IPAdapter due to NeRF architecture
    model_type = type(flux_model.diffusion_model).__name__
    if model_type == "ChromaRadiance":
        print("WARNING: ChromaRadiance models are not compatible with IPAdapter due to their NeRF-based architecture.")
        print("IPAdapter will be disabled for this model. The model will run normally without IPAdapter effects.")
        return  # Skip patching for ChromaRadiance
    
    bi.add_object_patch(f"diffusion_model.forward_orig", MethodType(forward_orig_ipa, flux_model.diffusion_model))
    for i, original in enumerate(flux_model.diffusion_model.double_blocks):
        patch_name = f"double_blocks.{i}"
        maybe_patched_layer = bi.get_model_object(f"diffusion_model.{patch_name}")
        # if there's already a patch there, collect its adapters and replace it
        procs = [ip_attn_procs[patch_name]]
        embs = [image_emb]
        if isinstance(maybe_patched_layer, DoubleStreamBlockIPA):
            procs = maybe_patched_layer.ip_adapter + procs
            embs = maybe_patched_layer.image_emb + embs
        # initial ipa models with image embeddings
        new_layer = DoubleStreamBlockIPA(original, procs, embs)
        # TODO: maybe there's a different patching method that will automatically chain patches?
        # for example, ComfyUI internally uses model.add_patches to add loras
        bi.add_object_patch(f"diffusion_model.{patch_name}", new_layer)
    for i, original in enumerate(flux_model.diffusion_model.single_blocks):
        patch_name = f"single_blocks.{i}"
        maybe_patched_layer = bi.get_model_object(f"diffusion_model.{patch_name}")
        procs = [ip_attn_procs[patch_name]]
        embs = [image_emb]
        if isinstance(maybe_patched_layer, SingleStreamBlockIPA):
            procs = maybe_patched_layer.ip_adapter + procs
            embs = maybe_patched_layer.image_emb + embs
        # initial ipa models with image embeddings
        new_layer = SingleStreamBlockIPA(original, procs, embs)
        bi.add_object_patch(f"diffusion_model.{patch_name}", new_layer)
        
def is_model_patched(model):
    def test(mod):
        if isinstance(mod, DoubleStreamBlockIPA):
            return True
        else:
            for p in mod.children():
                if test(p):
                    return True
        return False
    result = test(model)
    return result

def forward_orig_ipa(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor|None = None,
    control=None,
    transformer_options={},
    attn_mask: Tensor = None,
) -> Tensor:
    patches_replace = transformer_options.get("patches_replace", {})
    
    # running on sequences img
    img = self.img_in(img)
    
    # Check if this is a Chroma model (has distilled_guidance_layer instead of time_in)
    is_chroma = hasattr(self, 'distilled_guidance_layer') and not hasattr(self, 'time_in')
    
    if is_chroma:
        # Chroma model: use distilled guidance layer
        mod_index_length = 344
        distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(img.device, img.dtype)
        
        if guidance is None:
            guidance = torch.zeros_like(timesteps)
        distil_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.device, img.dtype)
        
        # get all modulation index
        modulation_index = timestep_embedding(torch.arange(mod_index_length, device=img.device), 32).to(img.device, img.dtype)
        # broadcast the modulation index
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1).to(img.device, img.dtype)
        # broadcast timestep and guidance
        timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1).to(img.dtype).to(img.device, img.dtype)
        # concatenate
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1).to(img.device, img.dtype)
        mod_vectors = self.distilled_guidance_layer(input_vec)
    else:
        # Standard Flux model: use time_in, guidance_in, and vector_in
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
        vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
    
    txt = self.txt_in(txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        # Check if we should skip this block (Chroma models only)
        if is_chroma and hasattr(self, 'skip_mmdit') and i in self.skip_mmdit:
            continue
            
        # Prepare vec for this block based on model type
        if is_chroma:
            # For Chroma: extract modulations for this specific block
            double_mod = (
                self.get_modulations(mod_vectors, "double_img", idx=i),
                self.get_modulations(mod_vectors, "double_txt", idx=i),
            )
            vec = double_mod
        # else vec is already set for standard Flux models
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                if isinstance(block, DoubleStreamBlockIPA): # ipadaper 
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], t=args["timesteps"], attn_mask=args.get("attn_mask"))
                else:
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"))
                return out
            out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "timesteps": timesteps, "attn_mask": attn_mask}, {"original_block": block_wrap,"transformer_options": transformer_options})
            txt = out["txt"]
            img = out["img"]
        else:
            if isinstance(block, DoubleStreamBlockIPA): # ipadaper 
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, t=timesteps, attn_mask=attn_mask)
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask)

        if control is not None: # Controlnet
            control_i = control.get("input")
            if control_i is not None and i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
        # Check if we should skip this block (Chroma models only)
        if is_chroma and hasattr(self, 'skip_dit') and i in self.skip_dit:
            continue
            
        # Prepare vec for this block based on model type
        if is_chroma:
            # For Chroma: extract modulations for this specific single block
            single_mod = self.get_modulations(mod_vectors, "single", idx=i)
            vec = single_mod
        # else vec is already set for standard Flux models
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                if isinstance(block, SingleStreamBlockIPA): # ipadaper
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], t=args["timesteps"], attn_mask=args.get("attn_mask"))
                else:
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask"))
                return out

            out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "timesteps": timesteps, "attn_mask": attn_mask}, {"original_block": block_wrap, "transformer_options": transformer_options})
            img = out["img"]
        else:
            if isinstance(block, SingleStreamBlockIPA): # ipadaper
                img = block(img, vec=vec, pe=pe, t=timesteps, attn_mask=attn_mask)
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

        if control is not None: # Controlnet
            control_o = control.get("output")
            if control_o is not None and i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1] :, ...] += add

    img = img[:, txt.shape[1] :, ...]

    # Prepare vec for final layer based on model type
    if is_chroma:
        # For Chroma: extract final modulations
        if hasattr(self, "final_layer"):
            final_mod = self.get_modulations(mod_vectors, "final")
            img = self.final_layer(img, vec=final_mod)  # (N, T, patch_size ** 2 * out_channels)
    else:
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img
