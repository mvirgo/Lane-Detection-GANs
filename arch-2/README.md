# Architecture 2
Check performance of the original network by adding in batch normalization for all layers, adding skip layers, and trying strides of 2 instead of max pooling.

## Summary
1. Attempted models with each of these individually (from base of `arch-1`)
   - `arch-2.1` - No pooling (replaced by strides of 2x2 over max pooling)
   - `arch-2.2` - Batch normalization on every conv layer
   - Adding skip layers
     - `arch-2.3` - skip over 2 conv layers every time (uses 1x1 convolutions to match sizes)
     - `arch-2.4` - skip over all in-between layers to the matching deconvolutional layer
   - Combinations of the above
     - `arch-2.5` - combine `arch-2.1` and `arch-2.2` (no pooling and add batch norm to all layers)
     - `arch-2.6` - `arch-2.3` along with removing pooling and adding batch norm (ala `arch-2.5`)
     - `arch-2.7` - `arch-2.4` along with adding batch norm (ala `arch-2.2`)

## Results

Arch | Test Acc | Speed | Parameters
--- | --- | --- | ---
arch-2.1 | 0.8667 | 4.82 ms | 725,101
arch-2.2 | 0.8677 | 14.55 ms | 728,749
arch-2.3 | 0.8683 | 6.18 ms | 746,413
**arch-2.4** | 0.8691 | 5.35 ms | 725,101
arch-2.5 | 0.8663 | 14.41 ms | 728,749
arch-2.6 | 0.8681 | 19.22 ms | 751,341
arch-2.7 | 0.8684 | 14.80 ms | 728,749

Interestingly here, the networks without skip layers work best with `mean_squared_error` as a loss function, while those with skip layers need `binary_crossentropy` as a loss function, or else they stop learning fairly early on.