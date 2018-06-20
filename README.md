# Lane-Detection-GANs
Investigating using GANS to help train a lane detection neural network

## Purpose
To investigate whether the usage of a GAN-trained generator can be used as the basis of weights for a semantic segmentation model for lane detection.

## Planned Architectures
1. Check performance of network similar to original network [here](https://github.com/mvirgo/MLND-Capstone) when using square images instead of rectangular, to fit with the norm in training convolutional neural networks.
2. Check performance of the network by adding in batch normalization for all layers, adding skip layers, and trying strides of 2 instead of max pooling.
3. Check performance gained when using an ImageNet pretrained architecture (Inception, ResNet, etc.) as the encoder part of the network, with a similar decoder structure as before for the semantic inference. Check both speed and inference accuracy.
4. Check performance if training the full architecture of an ImageNet pretrained architecture like #3, but without freezing the pretrained layers, and adding that to the original decoder structure.
5. Use a GAN to train a generator to produce fake semantic outputs, with the original real semantic outputs from the dataset as the "real" inputs. 
6. Using the optimal encoder network from steps 1-4, lead into a fully-connected layer that acts similar to the noise vector that the generator made in step 5 would typically start with. Using the optimal encoder plus the generator as decoder, check performance against the original (speed and accuracy). Likely will need to drop final layer of optimal encoder to train (so that it matches to the noise vector for pre-trained decoder), and then also train final layer of decoder. Can try both this method and full re-training with the pre-trained weights to start with.
7. Potentially trade back and forth between a fully convolutional model and the generator portion of a GAN, such that the "embeddings" in the middle of the fully convolutional model are similar to the generator's input noise, and go back and forth until little improvement is seen. 
8. Optimize for inference; again compare for speed and accuracy.

## Training
The `train_net.py` file is used to run training on a given neural network architecture within the `architectures` folder. In order to do so, you must specify the location of the model, as if importing it as a module:

`python train_net.py -a "architectures.arch-1.arch-1a"`

Only the model location needs to be specified, although you can also use the following:
+ `-a` - location of the model, as shown above. **Required**
+ `-b` - batch size; defaults to 16
+ `-e` - maximum number of epochs (script also contains `EarlyStopping`); defaults to 50
+ `-fa` - the activation function to be used on the output layer; defaults to `'sigmoid'`
+ `-l` - the loss function to be used; defaults to `'binary_crossentropy'`
+ `-m` - if using "Inception" or "ResNet" pretrained on ImageNet, specify here so proper pre-processing is used (optional)

For example, if desiring a batch size of 64, max epochs of 25, "relu" activation on the output, and "mean_squared_error" as the loss function, you'd use:

`python train_net.py -a "architectures.arch-1.arch-1a" -b 64 -e 25 -fa "relu" -l "mean_squared_error"`

Add `-m "resnet"` or `-m "inception"` to appropriately work with those models.

All models are trained with the default output activation and loss above, unless otherwise specified.

## Results
**In Progress**

**Results below need to be updated for arch-4 due to change in labeling**

**Upcoming Changes** - Based on testing the first four architectures, there actually is surprisingly little difference specific to lane detection. This may be due to the simplicity of the task, where the lane is nearly always in the same location, and it's only the edges of the lane where most changes take place. As such, I might consider re-doing the work on a simplified version of Cityscapes dataset (say on a subset of classes), to potentially get more useful results.

Inference times benchmarked using a GTX 1060.
See `performance.py` file for example evaluation script. This also takes the arguments `-a` and `-m` per above under **Training**, *however* here `-a` refers to the location of the trained `.h5` file, and should both include "/" directories as well as ".h5" at the end. Somewhat confusing, but I may update this to line up later.

Test dataset is based on annotations of Udacity's challenge video from Advanced Lane Finding.

Test accuracy is based on intersection over union metric.

Architecture | Test Acc | Speed | Parameters
--- | --- | --- | ---
arch-1 | 0.9891 | 5.47 ms | 725,101
arch-2 | 0.9920 | 6.24 ms | 746,413
arch-3 | 0.9705 | 42.18 ms | 26,233,441
arch-4 | 0.9764 | 83.43 ms | 25,441,345

`arch-1`, `arch-2` and `arch-4` are also fairly visually appealing, while `arch-3` seems to be more like a lane-shaped blob that goes outside the lane lines often.
