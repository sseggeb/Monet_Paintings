# I'm Something of a Painter Myself: Monet Style Transfer (CycleGAN)

This repository contains the TensorFlow implementation of a **Cycle Generative Adversarial Network (CycleGAN)** for the Kaggle competition, focusing on translating photographs into the distinctive artistic style of Claude Monet.

The model is designed to be highly optimized for execution on **Google's TPU v3-8 accelerator** within the Kaggle/Colab environment.

## Project Goal

The primary goal is to train a CycleGAN to learn the mapping function between a set of input photographs and a set of Monet paintings, without requiring paired training data. The result is a generator capable of transforming any new photograph into a Monet-esque image.

## Architecture and Technical Highlights

The solution is built upon the standard CycleGAN architecture, incorporating two Generators ($G_{Photo \to Monet}, G_{Monet \to Photo}$) and two Discriminators ($D_{Monet}, D_{Photo}$).

### Key Architectural Components

* **Generator (G):** Uses a **U-Net** architecture with skip connections to enable high-resolution image translation while preserving fine details.
* **Discriminator (D):** Uses a **PatchGAN** architecture, which classifies $32 \times 32$ patches of the image as real or fake, forcing the generator to focus on local texture and style details.

### Crucial Implementation Fixes (Overcome Challenges)

The final, stable notebook incorporates several critical fixes necessary for stable training on a TPU:

1.  **Modern Normalization Fix:** Replaced the deprecated `tensorflow_addons.InstanceNormalization` layer with the officially supported and equivalent **`tf.keras.layers.GroupNormalization(groups=1)`** in both the Generator and Discriminator networks. This was essential for training stability and preventing `NaN` losses.
2.  **Loss Scaling for PatchGAN on TPU:** Corrected the adversarial loss calculation to properly average the loss over the large $32 \times 32$ PatchGAN output elements. This was achieved by setting `reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` in the `BinaryCrossentropy` object.
3.  **TPU Initialization and Scope Management:** Ensured all models, optimizers, and the final `CycleGan` wrapper class were instantiated within the `tf.distribute.TPUStrategy` scope for distributed training across 8 TPU cores.
4.  **Accelerated Inference for Submission:** Optimized the final image generation step by using a large **Batch Size (e.g., 16 or 32)** and ensuring the forward pass ran entirely within the `strategy.scope()`, mitigating the risk of the session timing out due to slow CPU-bound generation.

## How to Run the Notebook

This notebook is designed to be run on Kaggle or Google Colab with the TPU enabled.

1.  **Setup Environment:**
    * Start a new Kaggle Notebook session.
    * Go to **Settings** and select **TPU v3-8** as the Accelerator.
2.  **Run TPU Setup:** Execute the first code cell to connect to the TPU cluster and ensure the output shows **`Number of replicas in sync: 8`**.
3.  **Data Loading:** The notebook uses the direct path for the Kaggle dataset: `'/kaggle/input/gan-getting-started'`.
4.  **Training:** The `cycle_gan_model.fit()` function will train the model for 25+ epochs. Checkpointing is implemented to allow resumption from a saved state.
5.  **Generate Submission:** After training, run the final cells to:
    * Display a side-by-side example of an input photo and the generated Monet-style image.
    * Generate and save the required 7,000+ output images, compressed into the final `images.zip` submission file.

