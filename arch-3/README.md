# Architecture 3
Check performance gained when using an ImageNet pretrained architecture (Inception, ResNet, etc.) as the encoder part of the network, with a similar decoder structure as before for the semantic inference.

**Note: The trained model files are not included within this repository due to file size >100 MB.**

## Summary
1. Used ResNet for the encoder portion of the architecture
   - `arch 3.1` - uses 1x1 convolution and `Flatten()` before fully-connected "middle" layer
   - `arch 3.3` - uses `GlobalAveragePooling2D` before fully-connected "middle" layer
2. Used Inception for the encoder portion of the architecture
   - `arch 3.2` - uses 1x1 convolution and `Flatten()` before fully-connected "middle" layer
   - `arch 3.4` - uses `GlobalAveragePooling2D` before fully-connected "middle" layer

## Results

Arch | Test Acc | Speed | Parameters
--- | --- | --- | ---
arch-3.1 | 0.8672 | 40.54 ms | 29,098,209
arch-3.2 | 0.8667 | 69.58 ms | 28,249,281
arch-3.3 | 0.8704 | 40.10 ms | 27,226,273
**arch-3.4** | 0.8712 | 67.86 ms | 25,441,345