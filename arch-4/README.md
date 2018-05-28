# Architecture 4
Similar to architecture 3, check performance gained when using an ImageNet pretrained architecture (Inception, ResNet, etc.) as the encoder part of the network, but without freezing layers, with a similar decoder structure as before for the semantic inference.

**Note: The trained model files are not included within this repository due to file size >100 MB.**

## Summary
1. Used ResNet for the encoder portion of the architecture
   - `arch-4.1` - uses `GlobalAveragePooling2D` before fully-connected "middle" layer
2. Used Inception for the encoder portion of the architecture
   - `arch-4.2` - uses `GlobalAveragePooling2D` before fully-connected "middle" layer

## Results

Arch | Test Acc | Speed | Parameters
--- | --- | --- | ---
arch-4.1 | 0.9751 | 48.60 ms | 27,226,273
**arch-4.2** | 0.9764 | 83.43 ms | 25,441,345

MSE with ReLU output was found to be more accurate than Binary Crossentropy with Sigmid output.