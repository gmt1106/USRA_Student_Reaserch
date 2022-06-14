# Image and video de-noising through neural fields 

In this project we aim to train a Neural Field network to denoise/upsample images and videos. In the realm of Convolutional Neural Networks, a method named Deep Image Prior has shown that one can successfully upsample and denoise images using this framework, without any training. Furthermore, recent works on Magnetic Resonance Imaging has also shown that this can be done in a spatio-temporal manner as well. In this project, we try to extend this finding to the realm of implicit representations, or more recently named, neural fields. With a neural field representation, one aims to train a deep network that maps spatial locations of a signal to its measurement. By doing so, one can then use this mapping to re-sample the data point at any resolution as desired. This method has shown tremendous success in the area of 3D vision, but has not yet been explored in the context of denoising and upsampling. We will first investigate the potential of this method, then extend it by including physical models of the noise in the data, as well as the downsampling process to further enhance its potential.


Source: 
- https://dmitryulyanov.github.io/deep_image_prior
- https://www.vincentsitzmann.com/siren/
- https://github.com/lucidrains/siren-pytorch
