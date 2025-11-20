# Chroma Model Compatibility Changes

This document details all modifications made to ComfyUI-IPAdapter-Flux to support Chroma models alongside standard Flux models.

## Overview

Chroma models use a fundamentally different architecture compared to standard Flux models, particularly in how they handle modulation and timestep embeddings. These changes enable the IPAdapter to work seamlessly with both model types by detecting which architecture is being used and adapting accordingly.

---

## Files Modified

### 1. `flux/layers.py`

#### Changes to `DoubleStreamBlockIPA.__init__`

**Problem:** Chroma's `DoubleStreamBlock` doesn't have `img_mod` and `txt_mod` attributes that exist in standard Flux models.

**Solution:** Added detection and conditional attribute assignment.

```python
# Check if this is a Chroma model (no img_mod/txt_mod attributes)
self.is_chroma = not hasattr(original_block, 'img_mod')

if not self.is_chroma:
    self.img_mod = original_block.img_mod
    self.txt_mod = original_block.txt_mod
```

**Impact:** The wrapper now only attempts to copy modulation attributes if they exist in the original block.

---

#### Changes to `DoubleStreamBlockIPA.forward`

**Problem:** Chroma models pass modulation values directly through the `vec` parameter as a tuple of `ChromaModulationOut` objects, while Flux models require processing `vec` through modulation layers.

**Solution:** Implemented dual-path logic based on model type.

**Modulation Handling:**
```python
if self.is_chroma:
    # Chroma: vec is already unpacked modulation values
    (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec
else:
    # Flux: vec needs processing through modulation layers
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)
```

**Image/Text Modulation Application:**
```python
# Image modulation
if self.is_chroma:
    img_modulated = torch.addcmul(img_mod1.shift, 1 + img_mod1.scale, img_modulated)
else:
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
```

**Output Calculation:**
```python
# Chroma uses in-place operations
if self.is_chroma:
    img.addcmul_(img_mod1.gate, self.img_attn.proj(img_attn))
    img.addcmul_(img_mod2.gate, self.img_mlp(torch.addcmul(...)))
else:
    img = img + img_mod1.gate * self.img_attn.proj(img_attn)
    img = img + img_mod2.gate * self.img_mlp(...)
```

**Impact:** The block now correctly handles modulation for both architectures, using `torch.addcmul` operations for Chroma and standard arithmetic for Flux.

---

#### Changes to `SingleStreamBlockIPA.__init__`

**Problem:** Chroma's `SingleStreamBlock` doesn't have a `modulation` attribute, and in ComfyUI v0.3.69+, `modulation` can also be `None` when `modulation=False` is passed to the constructor.

**Solution:** Added similar detection and conditional assignment with None check.

```python
# Check if this is a Chroma model (no modulation attribute or modulation is None)
self.is_chroma = not hasattr(original_block, 'modulation') or original_block.modulation is None
if not self.is_chroma:
    self.modulation = original_block.modulation
```

**Impact:** Handles both Chroma models and new ComfyUI versions where modulation can be disabled.

---

#### Changes to `SingleStreamBlockIPA.forward`

**Problem:** Similar modulation handling differences as in double stream blocks, with additional None check needed for ComfyUI v0.3.69+.

**Solution:** Implemented dual-path modulation processing with null safety.

```python
if self.is_chroma:
    mod = vec  # Already processed
    x_mod = torch.addcmul(mod.shift, 1 + mod.scale, self.pre_norm(x))
else:
    # Handle case where modulation might be None (ComfyUI v0.3.69+)
    if self.modulation is not None:
        mod, _ = self.modulation(vec)
    else:
        mod = vec
    x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
```

**Impact:** Prevents `TypeError: 'NoneType' object is not callable` when using ComfyUI v0.3.69+ where blocks can have `modulation=None`.

---

### 2. `utils.py`

#### Changes to `forward_orig_ipa` Function

**Problem 1:** Chroma models don't have `time_in`, `guidance_in`, or `vector_in` attributes. Instead, they use a `distilled_guidance_layer` for modulation.

**Solution:** Added model type detection and dual-path vec computation.

```python
# Check if this is a Chroma model
is_chroma = hasattr(self, 'distilled_guidance_layer') and not hasattr(self, 'time_in')

if is_chroma:
    # Chroma: use distilled guidance layer
    mod_index_length = 344
    distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(img.device, img.dtype)
    
    if guidance is None:
        guidance = torch.zeros_like(timesteps)
    distil_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.device, img.dtype)
    
    # Create modulation index and broadcast
    modulation_index = timestep_embedding(torch.arange(mod_index_length, device=img.device), 32)
    modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
    
    # Combine timestep, guidance, and modulation index
    timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1)
    timestep_guidance = timestep_guidance.unsqueeze(1).repeat(1, mod_index_length, 1)
    input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)
    
    # Generate all modulation vectors at once
    mod_vectors = self.distilled_guidance_layer(input_vec)
else:
    # Standard Flux: use time_in, guidance_in, vector_in
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
    vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
```

**Impact:** The function now generates the appropriate modulation vectors for each model type.

---

**Problem 2:** Chroma models use block-specific modulations extracted from a single vector, while Flux uses a single vec for all blocks.

**Solution:** Extract modulations per-block for Chroma models.

```python
for i, block in enumerate(self.double_blocks):
    # Skip blocks if needed (Chroma-specific)
    if is_chroma and hasattr(self, 'skip_mmdit') and i in self.skip_mmdit:
        continue
    
    # Prepare vec for this block
    if is_chroma:
        double_mod = (
            self.get_modulations(mod_vectors, "double_img", idx=i),
            self.get_modulations(mod_vectors, "double_txt", idx=i),
        )
        vec = double_mod
    # else vec is already set for Flux models
    
    # Process block...
```

**Impact:** Each block now receives the correct modulation format for its architecture.

---

**Problem 3:** Control inputs could be None, causing `len()` to fail.

**Solution:** Added None checks before accessing length.

```python
if control is not None:
    control_i = control.get("input")
    if control_i is not None and i < len(control_i):
        # Process control...
```

---

**Problem 4:** Chroma's final layer requires specific modulation format.

**Solution:** Extract final modulations correctly for Chroma.

```python
if is_chroma:
    if hasattr(self, "final_layer"):
        final_mod = self.get_modulations(mod_vectors, "final")
        img = self.final_layer(img, vec=final_mod)
else:
    img = self.final_layer(img, vec)
```

**Impact:** Final layer now works correctly for both model types.

---

**Problem 5:** Chroma models have `skip_mmdit` and `skip_dit` attributes to skip certain blocks during processing.

**Solution:** Added skip checks in block loops.

```python
# For double blocks
if is_chroma and hasattr(self, 'skip_mmdit') and i in self.skip_mmdit:
    continue

# For single blocks  
if is_chroma and hasattr(self, 'skip_dit') and i in self.skip_dit:
    continue
```

**Impact:** Respects Chroma's block skipping mechanism for proper model behavior.

---

### 3. `attention_processor_advanced.py`

#### Changes to `IPAFluxAttnProcessor2_0Advanced.clear_memory`

**Problem:** The `clear_memory()` method was deleting essential model parameters (`to_k_ip`, `to_v_ip`, `norm_added_k`, `norm_added_v`) between generations, causing the model to fail on subsequent generations.

**Solution:** Modified to only clear temporary caches, not model parameters.

```python
def clear_memory(self):
    """Clear temporary caches but keep model parameters intact"""
    self.seen_timesteps.clear()
    # Don't delete model parameters (to_k_ip, to_v_ip, norm_added_k, norm_added_v)
    # as they are nn.Module components needed for forward passes
```

**Impact:** Model now works correctly across multiple generations without losing its learned parameters.

---

## Technical Details

### Chroma vs Flux Architecture Differences

| Aspect | Flux | Chroma |
|--------|------|--------|
| **Timestep Embedding** | `time_in`, `guidance_in`, `vector_in` | `distilled_guidance_layer` |
| **Modulation Attributes** | `img_mod`, `txt_mod` in blocks | No modulation attributes in blocks |
| **Modulation Passing** | Single `vec` processed per block | All modulations pre-computed, indexed per block |
| **Modulation Format** | `ModulationOut` from layer | `ChromaModulationOut` from global vector |
| **Operations** | Standard arithmetic | In-place `torch.addcmul` operations |
| **Block Skipping** | None | `skip_mmdit`, `skip_dit` attributes |
| **Final Layer** | Simple vec passing | Requires "final" modulation type |

### Modulation Types in Chroma

Chroma's `get_modulations()` method supports these block types:

- `"single"` (idx required): Single stream block modulations
- `"double_img"` (idx required): Double stream image modulations  
- `"double_txt"` (idx required): Double stream text modulations
- `"final"` (no idx): Final layer modulations

The function returns tuples of `ChromaModulationOut` objects with `shift`, `scale`, and `gate` attributes.

---

## Testing

The modifications have been tested with:
- ✅ Standard Flux models
- ✅ Chroma distilled models
- ✅ Multiple generations in sequence
- ✅ Different IPAdapter weights and configurations
- ✅ Various sampling methods

---

## Backwards Compatibility

All changes maintain full backwards compatibility with standard Flux models. The detection mechanism automatically identifies the model type and uses the appropriate code path, requiring no user configuration or workflow changes.

---

## Future Considerations

1. **ChromaRadiance Support**: The current implementation should work with ChromaRadiance models, but may need verification.

2. **Performance Optimization**: The block-by-block modulation extraction for Chroma could potentially be optimized by pre-extracting all modulations once.

3. **Memory Management**: Monitor memory usage with Chroma's pre-computed modulation vectors for very large batch sizes.

---

## Credits

These modifications enable ComfyUI-IPAdapter-Flux to work with Black Forest Labs' Chroma model architecture while maintaining compatibility with the original Flux models.

**Date:** November 18, 2025
**Modified Files:** 3 files, ~200 lines of changes
**Compatibility:** Flux, Chroma (and likely ChromaRadiance)
