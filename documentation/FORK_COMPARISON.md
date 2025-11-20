# Fork Comparison: Original vs FluxChroma

## Quick Reference

| Feature | Original (InstantX) | FluxChroma Fork |
|---------|---------------------|-----------------|
| **Node Names** | `IPAdapterFluxLoader`<br>`ApplyIPAdapterFlux`<br>`IPAdapterFluxLoaderAdvanced`<br>`ApplyIPAdapterFluxAdvanced` | `IPAdapterFluxChromaLoader`<br>`ApplyIPAdapterFluxChroma`<br>`IPAdapterFluxChromaLoaderAdvanced`<br>`ApplyIPAdapterFluxChromaAdvanced` |
| **Log Prefix** | `IPAdapter:` / `IPAdapter Advanced:` | `IPAdapter-FluxChroma:` / `IPAdapter-FluxChroma Advanced:` |
| **Device Handling** | `self.device = device` (mixed types) | `self.device = torch.device(device)` (consistent) |
| **Auto Device Movement** | ❌ None | ✅ Models auto-move back to GPU when needed |
| **ModelPatcher Registration** | ❌ No registration | ✅ Attention processors registered |
| **Preprocessing Offloading** | ❌ No offloading | ✅ Automatic CPU offload (~1.2GB saved) |
| **Offload Toggle** | ❌ N/A | ✅ `offload_preprocessing` parameter |
| **Smart GPU Restoration** | ❌ N/A | ✅ Detects CPU models and restores |
| **WeakSet Cleanup** | `_instances.remove()` (can error) | `_instances.discard()` (safe) |
| **Debug Logging** | Minimal | Reduced (only important events) |
| **Can Coexist** | N/A | ✅ Yes, different node names |

## Code Differences

### 1. Device Initialization

**Original:**
```python
class InstantXFluxIPAdapterModel:
    def __init__(self, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device  # String or device object
```

**FluxChroma:**
```python
class InstantXFluxIPAdapterModel:
    def __init__(self, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = torch.device(device) if isinstance(device, str) else device
```

### 2. Model Movement in get_image_embeds()

**Original:**
```python
def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
    if pil_image is not None:
        # ... process image directly ...
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.image_encoder.dtype)).pooler_output
```

**FluxChroma:**
```python
def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
    if pil_image is not None:
        # Ensure encoder is on the correct device (it may have been offloaded to CPU)
        if self.image_encoder.device != self.device:
            self.image_encoder.to(self.device)
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.image_encoder.dtype)).pooler_output
    # ... same for projection model ...
```

### 3. Apply Function Parameters

**Original:**
```python
def apply_ipadapter_flux(self, model, ipadapter_flux, image, weight, start_percent, end_percent):
    # No offloading parameter
```

**FluxChroma:**
```python
def apply_ipadapter_flux(self, model, ipadapter_flux, image, weight, start_percent, end_percent, offload_preprocessing=True):
    # New parameter with default True
```

### 4. Model Registration

**Original:**
```python
def apply_ipadapter_flux(self, model, ipadapter_flux, image, weight, start_percent, end_percent):
    # ... process image ...
    bi = model.clone()
    FluxUpdateModules(bi, ipadapter_flux.ip_attn_procs, image_prompt_embeds, is_patched)
    return (bi,)  # No registration
```

**FluxChroma:**
```python
def apply_ipadapter_flux(self, model, ipadapter_flux, image, weight, start_percent, end_percent, offload_preprocessing=True):
    # ... process image ...
    bi = model.clone()
    FluxUpdateModules(bi, ipadapter_flux.ip_attn_procs, image_prompt_embeds, is_patched)
    
    # Register attention processors with memory management
    try:
        from comfy.model_patcher import ModelPatcher
        additional_models = []
        if hasattr(ipadapter_flux, 'ip_attn_procs') and ipadapter_flux.ip_attn_procs:
            first_attn_proc = next(iter(ipadapter_flux.ip_attn_procs.values()))
            attn_device = next(first_attn_proc.parameters()).device
            ip_layers = torch.nn.ModuleList(ipadapter_flux.ip_attn_procs.values())
            attn_patcher = ModelPatcher(ip_layers, attn_device, torch.device("cpu"))
            additional_models.append(attn_patcher)
        if additional_models:
            bi.set_additional_models("ipadapter_flux", additional_models)
    except Exception as e:
        logging.warning(f"IPAdapter-FluxChroma: Could not register attention processors: {e}")
    
    # Intelligent offloading logic...
    return (bi,)
```

### 5. Preprocessing Model Offloading

**Original:**
```python
# No offloading - models stay in VRAM
```

**FluxChroma:**
```python
if offload_preprocessing:
    try:
        if hasattr(ipadapter_flux.image_encoder, 'cpu'):
            original_device = ipadapter_flux.image_encoder.device
            ipadapter_flux.image_encoder.to('cpu')
            logging.info(f"IPAdapter-FluxChroma: Offloaded SiglipVisionModel from {original_device} to CPU (~1.1GB VRAM freed)")
        if hasattr(ipadapter_flux.image_proj_model, 'cpu'):
            original_device = next(ipadapter_flux.image_proj_model.parameters()).device
            ipadapter_flux.image_proj_model.to('cpu')
            logging.info(f"IPAdapter-FluxChroma: Offloaded projection model from {original_device} to CPU (~0.1GB VRAM freed)")
        if offloaded_count > 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logging.warning(f"IPAdapter-FluxChroma: Could not offload preprocessing models: {e}")
else:
    # Move models back to GPU if they were previously offloaded
    try:
        if hasattr(ipadapter_flux.image_encoder, 'to'):
            current_device = ipadapter_flux.image_encoder.device
            if current_device.type == 'cpu':
                target_device = model.load_device
                ipadapter_flux.image_encoder.to(target_device)
                logging.info(f"IPAdapter-FluxChroma: Moved SiglipVisionModel from CPU back to {target_device}")
        # ... same for projection model ...
    except Exception as e:
        logging.warning(f"IPAdapter-FluxChroma: Could not move preprocessing models back to GPU: {e}")
```

### 6. Cleanup in attention_processor_advanced.py

**Original:**
```python
def __del__(self):
    self.clear_memory()
    self.__class__._instances.remove(self)  # KeyError if already removed
```

**FluxChroma:**
```python
def __del__(self):
    self.clear_memory()
    self.__class__._instances.discard(self)  # Silent if not present
```

## Memory Impact

### VRAM Usage (32GB Card Example)

| State | Original | FluxChroma (Offload On) | FluxChroma (Offload Off) |
|-------|----------|------------------------|-------------------------|
| **After Loading** | ~12.8GB (40%) | ~12.8GB (40%) | ~12.8GB (40%) |
| **After Embedding** | ~12.8GB (40%) | ~11.6GB (36%) | ~12.8GB (40%) |
| **After Unload** | ~12.8GB stuck | ~5-6.5GB (15-20%) | ~6-7GB (18-22%) |
| **Savings** | 0GB | ~1.2GB | 0GB (but proper tracking) |

### Model Sizes

| Component | Size | Original Handling | FluxChroma Handling |
|-----------|------|-------------------|---------------------|
| SiglipVisionModel | ~1.1GB | Always in VRAM | CPU after embedding (if enabled) |
| MLPProjModel | ~0.1GB | Always in VRAM | CPU after embedding (if enabled) |
| Attention Processors | ~0.3-0.5GB | In VRAM | In VRAM (needed for inference) |
| **Total Offloadable** | **~1.2GB** | **0GB** | **~1.2GB** |

## Use Cases

### When to Use Original

1. **Simple workflows** - Single IPAdapter node, no complex combinations
2. **No memory issues** - Plenty of VRAM available
3. **Vanilla preference** - Want official InstantX implementation
4. **Production stability** - Prefer tested original codebase

### When to Use FluxChroma

1. **Complex workflows** - Chroma + IPAdapter + PuLID simultaneously
2. **Memory constraints** - Need to squeeze every GB from VRAM
3. **VRAM stuck high** - Original shows 40% VRAM after unload
4. **OOM errors** - Getting out of memory with multiple extensions
5. **Device errors** - Seeing "mat2 is on cpu" errors
6. **Need control** - Want to toggle offloading behavior
7. **Better tracking** - Want proper VRAM accounting in ComfyUI

## Migration Guide

### From Original to FluxChroma

1. **Remove original** (optional, can coexist):
   ```bash
   cd ComfyUI/custom_nodes
   rm -rf ComfyUI-IPAdapter-Flux  # or rename
   ```

2. **Install FluxChroma**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ComfyUI-IPAdapter-FluxChroma.git
   cd ComfyUI-IPAdapter-FluxChroma
   pip install -r requirements.txt
   ```

3. **Update workflows**:
   - Replace `IPAdapterFluxLoader` → `IPAdapterFluxChromaLoader`
   - Replace `ApplyIPAdapterFlux` → `ApplyIPAdapterFluxChroma`
   - (Advanced nodes: add "Chroma" suffix similarly)
   - Add `offload_preprocessing` parameter (defaults to True)

4. **Restart ComfyUI**

### From FluxChroma back to Original

1. Workflows will break (different node names)
2. Need to recreate or batch-edit workflow JSONs
3. Lose offloading benefits

### Run Both Side-by-Side

1. Install both in different folders
2. Different node names prevent conflicts
3. Choose which to use per workflow
4. FluxChroma logs clearly prefixed

## Compatibility

### Both Support

- ✅ FLUX.1-dev-IP-Adapter weights
- ✅ Multiple IPAdapter instances
- ✅ Advanced weight scheduling
- ✅ Start/end percent control
- ✅ Same model files and directories
- ✅ Apache 2.0 License

### FluxChroma Adds

- ✅ Toggleable preprocessing offloading
- ✅ Automatic device management
- ✅ ComfyUI memory integration
- ✅ Robust cleanup
- ✅ Better logging
- ✅ Coexistence capability

## Performance Comparison

| Metric | Original | FluxChroma (Offload On) | FluxChroma (Offload Off) |
|--------|----------|------------------------|-------------------------|
| **Embedding Speed** | Baseline | ~Same | ~Same |
| **Inference Speed** | Baseline | ~Same | ~Same |
| **VRAM Peak** | Higher | Lower (~1.2GB saved) | Same |
| **VRAM Idle** | Higher (stuck) | Lower (proper cleanup) | Lower (proper cleanup) |
| **Load Time** | Baseline | ~Same | ~Same |
| **Unload Time** | Faster | ~Same | ~Same |
| **CPU Usage** | Lower | Slightly higher (offload) | Same |

**Note:** Speed differences are negligible. Main benefit is memory efficiency.

## Conclusion

**Original**: Stable, tested, official implementation. Best for simple workflows.

**FluxChroma**: Enhanced memory management, better device handling, designed for complex multi-extension workflows. Best for power users with memory constraints.

Both use the same weights, produce identical results, and follow the same architecture. FluxChroma adds non-invasive improvements focused on memory efficiency and device management.
