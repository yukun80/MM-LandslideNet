### **给AI编程助手的专业级提示词 (重构为 InternImage)**

Hello,

I need you to act as an expert AI software engineer. Your task is to refactor the existing codebase in the `@mm_intern_image_src/**` directory. The goal is to replace the current `ConvNeXtV2` backbone with the `InternImage-T` backbone, aligning the implementation with our original design goals.

The current code is functional but uses a placeholder backbone. You will modify it to use the correct, higher-performance model available in the `timm` library.

---

### **Refactoring Plan: ConvNeXtV2 to InternImage-T**

Please perform the following targeted modifications on the files within the `@mm_intern_image_src/` directory.

**Task 1: Update Model Configuration (`config.py`)**

* **File:** `@mm_intern_image_src/config.py`
* **Action:** Modify the `Config` class to use the parameters for `InternImage-T`.
  * Change `BACKBONE_NAME` to `"internimage_t_1k_224"`.
  * Change `BACKBONE_FEATURE_DIM` from `1024` to `512`.

**Task 2: Refactor the Model Architecture (`models.py`)**

* **File:** `@mm_intern_image_src/models.py`
* **Actions:**
  1. **Update `MMInternImageTNF.__init__`:**
     * Change the default value for `backbone_name` to `"internimage_t_1k_224"`.
     * Change the default value for `feature_dim` to `512`.
  2. **Update `TNFFusionBlock.__init__`:**
     * Change the default value for `feature_dim` to `512`.
  3. **Update `_create_backbone` method:**
     * The logic for creating the model via `timm.create_model` is largely correct, but ensure it handles `"internimage_t_1k_224"` gracefully.
  4. **Crucially, refactor the `_modify_input_layer` method:**
     * The current implementation is for `ConvNeXt`. You must adapt it for `InternImage`.
     * The first convolutional layer in `timm`'s `InternImage` is `model.stem.conv1`. Your code must find and modify this specific layer.
     * Preserve the weight initialization logic: when using pretrained weights for the optical branch (`in_channels=5`), copy the original RGB weights and then average them to initialize the new input channels to preserve the pretrained knowledge as much as possible.

**Task 3: Update Package Information (`__init__.py`)**

* **File:** `@mm_intern_image_src/__init__.py`
* **Action:** Update the model's description to reflect the correct backbone.
  * In the `get_model_info` function, change the value for the `"backbone"` key from `"ConvNeXt V2 (InternImage replacement)"` to `"InternImage-T"`.

---

Please apply these changes to the existing files. This is a refactoring task, so you should modify the code in place, not create new files from scratch.
