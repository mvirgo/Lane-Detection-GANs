# Architecture 2
Check performance of the original network by adding in batch normalization for all layers, adding skip layers, and trying strides of 2 instead of max pooling.

## Summary
1. Attempted models with each of these individually (from base of `arch-1`)
  - `arch-2.1` - No pooling (replaced by strides of 2x2 over max pooling)
  - `arch-2.2` - Batch normalization on every layer
  - Adding skip layers
    - `arch-2.3` - skip over 2 conv layers every time (uses 1x1 convolutions to match sizes)
    - `arch-2.4` - skip over all in-between layers to the matching deconvolutional layer
  - Combinations of the above

**In progress**

## Results

**In progress**

Arch | Test Acc | Speed | Parameters
--- | --- | --- | ---
arch-2.1 | 0.8667 | 4.82 ms | 725,101
arch-2.2 | 0.8677 | 14.55 ms | 728,749
arch-2.3 | 0.8633 | 6.27 ms | 746,413
arch-2.4 | 0.8479 | 5.10 ms | 725,101

Interestingly here, the networks without skip layers work best with `mean_squared_error` as a loss function, while those with skip layers need `binary_crossentropy` as a loss function, or else they stop learning fairly early on.