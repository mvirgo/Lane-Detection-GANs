# Architecture 2
Check performance of the network by adding in batch normalization for all layers, adding skip layers, and trying strides of 2 instead of max pooling.

## Summary
1. Attempted models with each of these individually
  - No pooling (replaced by strides of 2x2 over max pooling)
  - Batch normalization on every layer
  - Adding skip layers
    - Attempt 1 - skip over 2 conv layers every time
    - Attempt 2 - skip over all in-between layers to the matching deconvolutional layer
  - Combination of the above

**In progress**

## Results

**In progress**

Arch | Test Acc | Speed | Parameters
--- | --- | --- | ---
No pooling | 0.8683 | 4.75 ms | 725,101
Batch norm | 0.8692 | 12.04 ms | 728,749