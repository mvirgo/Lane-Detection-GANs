# Architecture 3
Check performance gained when using an ImageNet pretrained architecture (Inception, ResNet, etc.) as the encoder part of the network, with all layers frozen, with a similar decoder structure as before for the semantic inference.

**Note: The trained model files are not included within this repository due to file size >100 MB.**

## Summary
1. Used ResNet for the encoder portion of the architecture
   - `arch-3.1` - uses 1x1 convolution and `Flatten()` before fully-connected "middle" layer
   - `arch-3.3` - uses `GlobalAveragePooling2D` before fully-connected "middle" layer
   - `arch-3.5` - uses `ZeroPadding2D` to become 10x10 before using the previous decoder structure
2. Used Inception for the encoder portion of the architecture
   - `arch-3.2` - uses 1x1 convolution and `Flatten()` before fully-connected "middle" layer
   - `arch-3.4` - uses `GlobalAveragePooling2D` before fully-connected "middle" layer
   - `arch-3.6` - uses `ZeroPadding2D` to become 10x10 before using the previous decoder structure

## Results

Arch | Test Acc | Speed | Parameters
--- | --- | --- | ---
arch-3.1 | 0.9131 | 40.00 ms | 29,098,209
arch-3.2 | 0.9151 | 67.55 ms | 28,249,281
arch-3.3 | 0.9357 | 39.19 ms | 27,226,273
arch-3.4 | 0.9418 | 66.50 ms | 25,441,345
**arch-3.5** | 0.9460 | 39.75 ms | 26,233,441
arch-3.6 | 0.9338 | 66.94 ms | 24,448,513