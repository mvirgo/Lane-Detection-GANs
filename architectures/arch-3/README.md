# Architecture 3
Check performance gained when using an ImageNet pretrained architecture (Inception, ResNet, etc.) as the encoder part of the network, with all layers frozen, with a similar decoder structure as before for the semantic inference.

**Note: The trained model files are not included within this repository due to file size >100 MB.**

## Summary
1. Used ResNet for the encoder portion of the architecture
   - `arch-3a` - uses 1x1 convolution and `Flatten()` before fully-connected "middle" layer
   - `arch-3c` - uses `GlobalAveragePooling2D` before fully-connected "middle" layer
   - `arch-3e` - uses `ZeroPadding2D` to become 10x10 before using the previous decoder structure
2. Used Inception for the encoder portion of the architecture
   - `arch-3b` - uses 1x1 convolution and `Flatten()` before fully-connected "middle" layer
   - `arch-3d` - uses `GlobalAveragePooling2D` before fully-connected "middle" layer
   - `arch-3f` - uses `ZeroPadding2D` to become 10x10 before using the previous decoder structure

## Results

Arch | Test Acc | Speed | Parameters
--- | --- | --- | ---
arch-3a | 0.9649 | 43.26 ms | 29,098,209
arch-3b | 0.9595 | 71.06 ms | 28,249,281
arch-3c | 0.9565 | 40.85 ms | 27,226,273
arch-3d | 0.9486 | 70.24 ms | 25,441,345
**arch-3e** | 0.9705 | 42.18 ms | 26,233,441
arch-3f | 0.9694 | 70.28 ms | 24,448,513