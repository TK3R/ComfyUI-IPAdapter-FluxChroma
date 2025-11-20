# ComfyUI-IPAdapter-FluxChroma - Memory Management & Compatibility Improvements

## Fork Information

**Repository:** ComfyUI-IPAdapter-FluxChroma  
**Based on:** InstantX/ComfyUI-IPAdapter-Flux  
**Fork Date:** November 19-20, 2025  
**Purpose:** Enhanced memory management for Chroma + IPAdapter + PuLID workflows

---

## Complete Change Summary

### 1. Node Naming (Conflict Prevention)

**Changed to avoid conflicts with original extension:**

| Original | FluxChroma Fork |
|----------|----------------|
| `IPAdapterFluxLoader` | `IPAdapterFluxChromaLoader` |
| `ApplyIPAdapterFlux` | `ApplyIPAdapterFluxChroma` |
| `IPAdapterFluxLoaderAdvanced` | `IPAdapterFluxChromaLoaderAdvanced` |
| `ApplyIPAdapterFluxAdvanced` | `ApplyIPAdapterFluxChromaAdvanced` |

**Display Names:**
- "Load IPAdapter Flux Model (Chroma)"
- "Apply IPAdapter Flux Model (Chroma)"
- "Load IPAdapter Flux Model Advanced (Chroma)"
- "Apply IPAdapter Flux Model Advanced (Chroma)"

### 2. Device Handling Improvements

**Original Code:**
```python
self.device = device  # String or device object
```

**FluxChroma Code:**
```python
self.device = torch.device(device) if isinstance(device, str) else device
```

**Benefit:** Consistent torch.device objects prevent type mismatches and enable proper device.type checks.

### 3. Automatic Model Movement

**Added in get_image_embeds():**
```python
# Ensure encoder is on the correct device (it may have been offloaded to CPU)
if self.image_encoder.device != self.device:
    self.image_encoder.to(self.device)

# Ensure projection model is on the correct device
proj_device = next(self.image_proj_model.parameters()).device
if proj_device != self.device:
    self.image_proj_model.to(self.device)
```

**Benefit:** Models automatically move back to GPU when needed, even after being offloaded to CPU.

---

## Problem Identified (Original Extension)

IPAdapter Flux was not properly registering its models (image encoder and projection model) with ComfyUI's memory management system. This caused:

- **Incorrect VRAM calculations** - The memory management system didn't account for IPAdapter's models
- **No proper model loading/unloading** - Models stayed loaded in VRAM unnecessarily
- **Out of Memory (OOM) errors** - Especially when using with Chroma models and PuLID simultaneously
- **Inefficient VRAM usage** - Models couldn't be offloaded when needed

## Root Cause

The original implementation in `ipadapter_flux.py` used IPAdapter models without registering them:

```python
def apply_ipadapter_flux(self, model, ipadapter_flux, image, weight, start_percent, end_percent):
    # ... process image ...
    bi = model.clone()
    FluxUpdateModules(bi, ipadapter_flux.ip_attn_procs, image_prompt_embeds, is_patched)
    return (bi,)  # IPAdapter models not registered!
```

The `ipadapter_flux.image_encoder` (SiglipVisionModel), `ipadapter_flux.image_proj_model` (MLPProjModel), and `ipadapter_flux.ip_attn_procs` (57 attention processor modules) were loaded into VRAM but never tracked by ComfyUI's memory management.

## Solution Implemented

### Changes to `ipadapter_flux.py`:

Added model registration after `FluxUpdateModules()`:

```python
# Register IPAdapter models with memory management for proper VRAM tracking
try:
    from comfy.model_patcher import ModelPatcher
    additional_models = []
    
    # Create wrapper for image encoder
    encoder_device = ipadapter_flux.image_encoder.device
    encoder_offload = torch.device("cpu")
    encoder_patcher = ModelPatcher(ipadapter_flux.image_encoder, encoder_device, encoder_offload)
    additional_models.append(encoder_patcher)
    
    # Create wrapper for projection model
    proj_device = next(ipadapter_flux.image_proj_model.parameters()).device
    proj_patcher = ModelPatcher(ipadapter_flux.image_proj_model, proj_device, encoder_offload)
    additional_models.append(proj_patcher)
    
    # Create wrapper for attention processors (57 modules)
    ip_layers = torch.nn.ModuleList(ipadapter_flux.ip_attn_procs.values())
    attn_patcher = ModelPatcher(ip_layers, attn_device, encoder_offload)
    additional_models.append(attn_patcher)
    
    # Register all with the model
    bi.set_additional_models("ipadapter_flux", additional_models)
except Exception as e:
    logging.warning(f"Could not register IPAdapter models with memory management: {e}")
```

### Intelligent Offloading

After generating image embeddings, preprocessing models are automatically offloaded:

```python
# After embeddings are generated, offload preprocessing models
ipadapter_flux.image_encoder.to('cpu')  # Free ~1.1GB VRAM
ipadapter_flux.image_proj_model.to('cpu')  # Free ~0.1GB VRAM
torch.cuda.empty_cache()
```

Only the attention processors (~300-500MB) remain in VRAM for inference.

**Note about HuggingFace Models:**
- `image_encoder` (SiglipVisionModel) and `image_proj_model` NOT registered with ModelPatcher
- These models have read-only device properties incompatible with ComfyUI's automatic management
- Handled manually via explicit `.to()` calls instead
- Only attention processors are registered as they're standard PyTorch modules

---

## 4. Toggleable Offloading (November 20, 2025)

**New Parameter Added:**
```python
"offload_preprocessing": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"})
```

**Functionality:**

**When Enabled (default):**
```python
if offload_preprocessing:
    # Offload to CPU (saves ~1.2GB)
    ipadapter_flux.image_encoder.to('cpu')
    ipadapter_flux.image_proj_model.to('cpu')
    torch.cuda.empty_cache()
```

**When Disabled:**
```python
else:
    # Check if models are on CPU and move back to GPU if needed
    if ipadapter_flux.image_encoder.device.type == 'cpu':
        ipadapter_flux.image_encoder.to(model.load_device)
    if current_device.type == 'cpu':
        ipadapter_flux.image_proj_model.to(model.load_device)
```

**Benefits:**
- Default saves ~1.2GB VRAM for most workflows
- Flexibility for workflows that reuse IPAdapter multiple times
- Prevents "mat2 is on cpu" device mismatch errors when toggling
- Maintains backward compatibility

---

## 5. Logging Improvements

**Log Prefix Changes:**
- Original: `IPAdapter:` / `IPAdapter Advanced:`
- FluxChroma: `IPAdapter-FluxChroma:` / `IPAdapter-FluxChroma Advanced:`

**Reduced Log Verbosity:**
- Removed debug logs for routine operations (weight updates, embedding generation, registration)
- Kept important messages: offloading events, device movements, warnings
- Cleaner console output during normal operation

**Example Logs:**
```
[INFO] IPAdapter-FluxChroma: Offloaded SiglipVisionModel from cuda:0 to CPU (~1.1GB VRAM freed)
[INFO] IPAdapter-FluxChroma: Offloaded projection model from cuda:0 to CPU (~0.1GB VRAM freed)
[INFO] IPAdapter-FluxChroma: Moved SiglipVisionModel from CPU back to cuda:0
[WARNING] IPAdapter-FluxChroma: Could not register attention processors: <error>
```

---

## 6. Cleanup Robustness

**Fixed WeakSet KeyError:**

**Original Code:**
```python
def __del__(self):
    self.clear_memory()
    self.__class__._instances.remove(self)  # Could raise KeyError
```

**FluxChroma Code:**
```python
def __del__(self):
    self.clear_memory()
    self.__class__._instances.discard(self)  # Silent if not present
```

**Location:** `attention_processor_advanced.py`  
**Benefit:** Clean shutdown without exception spam during model unloading

---

## Summary of All Changes

### Code Changes

1. **Device handling** - Consistent `torch.device()` conversion
2. **Automatic device movement** - Models move back to GPU when needed
3. **ModelPatcher registration** - Attention processors tracked by ComfyUI
4. **Intelligent offloading** - Preprocessing models offload after use (toggleable)
5. **Smart GPU restoration** - Detects CPU models and restores to GPU
6. **Robust cleanup** - WeakSet.discard() instead of remove()
7. **Node renaming** - All nodes include "Chroma" suffix
8. **Reduced logging** - Only important messages logged
9. **Log prefixes** - Clear identification as FluxChroma fork

### Benefits

✅ **Proper VRAM tracking** - ComfyUI now accounts for attention processor memory  
✅ **Intelligent offloading** - Preprocessing models offload after use, saving ~1.2GB  
✅ **Reduced inference VRAM** - Only attention processors stay in VRAM during sampling  
✅ **Works with Chroma + IPAdapter + PuLID** - Multiple components coexist without OOM  
✅ **Better performance** - Memory managed based on actual requirements  
✅ **User control** - Toggle offloading for different workflow patterns  
✅ **Device safety** - Prevents device mismatch errors  
✅ **Clean shutdown** - No exception spam during unload  
✅ **No conflicts** - Can coexist with original extension  
✅ **Backward compatible** - Falls back gracefully if registration fails  

### File Modifications

**Modified Files:**
- `ipadapter_flux.py` - Basic node with all improvements
- `ipadapter_flux_advanced.py` - Advanced node with weight scheduling + improvements
- `attention_processor_advanced.py` - WeakSet cleanup fix
- `README.md` - Fork information and installation instructions
- `documentation/COMFY_MEMORY_MANAGEMENT_FIX.md` - This comprehensive document

**Unchanged from Original:**
- `attention_processor.py` - Basic attention processor (no changes needed)
- `utils.py` - Helper functions
- `flux/` - Model architecture
- `requirements.txt` - Dependencies
- `LICENSE` - Apache 2.0

---

## Original Documentation Context (Pre-Fork)

The sections below describe the original memory management approach that was enhanced in this fork.

## Technical Details (Original Implementation)

The fix wraps three critical IPAdapter components in `ModelPatcher`:

The fix wraps three critical IPAdapter components in `ModelPatcher`:

1. **SiglipVisionModel (image_encoder)**:
   - ~1.1GB VRAM for google/siglip-so400m-patch14-384
   - Now properly tracked and can be offloaded

2. **MLPProjModel (image_proj_model)**:
   - Variable size depending on num_tokens
   - Projects image features to cross-attention dimensions
   - Also now tracked and manageable

3. **Attention Processors (ip_attn_procs)**:
   - 57 IPAFluxAttnProcessor2_0 modules (19 double + 38 single blocks)
   - Each module has learnable parameters
   - Collectively can consume significant VRAM
   - Now wrapped in ModuleList and tracked

### Memory Management Integration

The fix integrates with ComfyUI's standard flow:

1. **During `_prepare_sampling()`**:
   - `model.get_nested_additional_models()` returns IPAdapter models
   - Memory calculations include image encoder and projection model
   - `load_models_gpu()` receives complete model list

2. **During `load_models_gpu()`**:
   - Total memory required includes IPAdapter components
   - `free_memory()` considers all models when deciding what to unload
   - `lowvram_model_memory` calculation is accurate

3. **During inference**:
   - Models are loaded/unloaded based on actual VRAM availability
   - Multiple extensions can share VRAM efficiently
   - Automatic offloading prevents OOM errors

## Verification

To verify the fix is working:

1. Check logs for registration warnings (should be none)
2. Monitor VRAM usage with IPAdapter active - should properly account for models
3. Test with Chroma + PuLID combinations - should work without OOM
4. Verify `model.get_nested_additional_models()` includes IPAdapter models

## Affected Models

The fix tracks these IPAdapter components:

- **Image Encoder**: SiglipVisionModel (typically ~1.1GB)
- **Projection Model**: MLPProjModel (variable size, depends on num_tokens)
- **Attention Processors**: 57 IPAFluxAttnProcessor2_0 modules (collectively ~200-500MB depending on configuration)
- **Note**: The attention processors are already patched into the model via FluxUpdateModules, but now they're also tracked for memory management

## Notes

- Uses try/except for backward compatibility
- Logs warnings if registration fails
- Does not change IPAdapter functionality, only adds memory management
- Compatible with existing workflows and configurations

## Recent Improvements (November 20, 2025)

### Toggleable Offloading

Added user control over preprocessing model offloading via `offload_preprocessing` boolean parameter:

**Features:**
- **Default: Enabled** - Maintains memory-saving behavior (~1.2GB VRAM freed)
- **User Control** - Can be disabled for workflows that reuse IPAdapter multiple times
- **Smart Device Management** - Automatically moves models back to GPU when toggling from enabled to disabled
- **Clear Logging** - Reports device movements and offload status

**Implementation:**
```python
# In INPUT_TYPES
"offload_preprocessing": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"})

# In apply function
if offload_preprocessing:
    # Offload to CPU (saves ~1.2GB)
    ipadapter_flux.image_encoder.to('cpu')
    ipadapter_flux.image_proj_model.to('cpu')
else:
    # Check if models are on CPU and move back to GPU if needed
    if ipadapter_flux.image_encoder.device.type == 'cpu':
        ipadapter_flux.image_encoder.to(model.load_device)
```

**Benefits:**
- Flexibility for different workflow patterns
- Prevents device mismatch errors when toggling
- Maintains backward compatibility with default enabled state

### Fixed WeakSet Cleanup Issue

Fixed KeyError during model unloading in attention processor cleanup:

**Problem:**
```python
# attention_processor_advanced.py line 62
def __del__(self):
    self.clear_memory()
    self.__class__._instances.remove(self)  # Could raise KeyError
```

**Solution:**
```python
def __del__(self):
    self.clear_memory()
    self.__class__._instances.discard(self)  # Silent if not present
```

Changed `remove()` to `discard()` to prevent KeyError when instances are removed multiple times or already cleaned up during garbage collection.

**Benefits:**
- Clean shutdown without exception spam
- Proper cleanup even in complex unloading scenarios
- More robust memory management

### Code Quality Fixes

**Indentation Errors (Fixed Nov 20):**
- Fixed try/except block indentation in both basic and advanced versions
- Added proper spacing before NODE_CLASS_MAPPINGS
- Ensured consistent 4-space indentation throughout offloading logic

**Device Handling:**
- Consistent use of `torch.device()` conversion
- In-place `.to()` operations for model movement
- Proper device type checking (`device.type == 'cpu'`)

These improvements ensure reliable memory management across all usage patterns and prevent common runtime errors.
