# Architecture 1
Check performance of a network similar to the original [here](https://github.com/mvirgo/MLND-Capstone) when using square images instead of rectangular, to fit with the norm in training convolutional neural networks.

## Summary
1. Re-made input images and labels to be 112x112x3. This is fairly close to the number of input parameters from the previous rectangular version
   - 112x112x3 = 37,632 input pixels
   - 80x160x3  = 38,400 input pixels previously
2. Using the exact same architecture as before
   - 2 Conv, max pooling, 3 Conv, max pooling, 2 Conv, max pooling for encoder
   - Upsampling, 2 DeConv, Upsampling, 3 DeConv, Upsampling, 2 DeConv
3. Tried two versions:
   - `arch-1.1` - Exact same as previous network, given the above
   - `arch-1.2` - Same as above, except change final layer to `sigmoid` activation and loss to `binary_crossentropy`

## Results

Arch | Test Acc | Speed | Parameters
--- | --- | --- | ---
**arch-1.1** | 0.8701 | 4.96 ms | 725,101
arch-1.2 | 0.8657 | 4.91 ms | 725,101