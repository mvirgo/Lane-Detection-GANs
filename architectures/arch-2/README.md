# Architecture 2
Check performance of the original network by adding in batch normalization for all layers, adding skip layers, and trying strides of 2 instead of max pooling.

## Summary
1. Attempted models with each of these individually (from base of `arch-1`)
   - `arch-2a` - No pooling (replaced by strides of 2x2 over max pooling)
   - `arch-2b` - Batch normalization on every conv layer
   - Adding skip layers
     - `arch-2c` - skip over 2 conv layers every time (uses 1x1 convolutions to match sizes)
     - `arch-2d` - skip over all in-between layers to the matching deconvolutional layer
   - Combinations of the above
     - `arch-2e` - combine `arch-2a` and `arch-2b` (no pooling and add batch norm to all layers)
     - `arch-2f` - `arch-2c` along with removing pooling and adding batch norm (ala `arch-2e`)
     - `arch-2g` - `arch-2d` along with adding batch norm (ala `arch-2b`)

## Results

Arch | Test Acc | Speed | Parameters
--- | --- | --- | ---
arch-2a | 0.9756 | 4.85 ms | 725,101
arch-2b | 0.9727 | 14.43 ms | 728,749
arch-2c | 0.9766 | 6.56 ms | 746,413
**arch-2d** | 0.9771 | 5.18 ms | 725,101
arch-2e | 0.9714 | 14.29 ms | 728,749
arch-2f | 0.9755 | 18.90 ms | 751,341
arch-2g | 0.9671 | 14.62 ms | 728,749

Interestingly here, the networks without skip layers work best with `mean_squared_error` as a loss function, while those with skip layers need `binary_crossentropy` as a loss function, or else they stop learning fairly early on.