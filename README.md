# Lane-Detection-GANs
Investigating using GANS to help train a lane detection neural network

## Purpose
To investigate whether the usage of a GAN-trained generator can be used as the basis of weights for a semantic segmentation model for lane detection.

## Planned Steps
1. Check performance of network similar to original network [here](https://github.com/mvirgo/MLND-Capstone) when using square images instead of rectangular, to fit with the norm in training convolutional neural networks.
2. Check performance of the network by adding in batch normalization for all layers, adding skip layers, and trying strides of 2 instead of max pooling.
3. Check performance gained when using an ImageNet pretrained architecture (MobileNet, Inception, ResNet, etc.) as the encoder part of the network, with a similar decoder structure as before for the semantic inference. Check both speed and inference accuracy.
4. Check performance if training just the original encoder portion on ImageNet, and adding that to the original decoder structure.
5. Use a GAN to train a generator to produce fake semantic outputs, with the original real semantic outputs from the dataset as the "real" inputs. 
6. Using the optimal encoder network from steps 1-4, lead into a fully-connected layer that acts similar to the noise vector that the generator made in step 5 would typically start with. Using the optimal encoder plus the generator as decoder, check performance against the original (speed and accuracy).
7. Potentially trade back and forth between a fully convolutional model and the generator portion of a GAN, such that the "embeddings" in the middle of the fully convolutional model are similar to the generator's input noise, and go back and forth until little improvement is seen. 
8. Optimize for inference; again compare for speed and accuracy.
