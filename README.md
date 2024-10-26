# Neural Monte Carlo Denoiser
Implementation of a neural network for denoising [Monte Carlo path-traced](https://en.wikipedia.org/wiki/Path_tracing) renders of virtual scenes from only single-sample inputs. Additionally contains dataset generation and loading scripts.

The main branch contains an implementation of the SIGGRAPH '21 paper [Interactive Monte Carlo Denoising using Affinity of Neural Features](https://www.mustafaisik.net/pdfs/isik2021-anf.pdf) by Isik et al. This uses a U-Net like architecture to predict neural features for each pixel from the noisy input radiance (demodulated point-wise by albedo) and auxiliary feature buffers, from which a set of four kernels are computed for each pixel and iteratively applied to the respective pixel neighborhoods. The fourth kernel is applied to the optical-flow warped output of the previous frame. 

This implementation additionally uses the optical-flow vectors to compute an exponential moving average over the model outputs, with geometry aware heuristics to prevent blending of uncorrelated pixels.

## Video

![1 Sample-per-pixel input vs. Denoised output](videos/comparison-anf.gif)

Note: the above GIF was heavily compressed to fit within the GitHub file size limits. Please see the [YouTube mirror of the same video](https://www.youtube.com/watch?v=_xkhUjZIMFE) for a higher quality render. All scenes depicted were **not** included in the training set.
