# Autoencoders
An autoencoder is a neural network that is trained to attempt to copy its input to its output. The basic architecture of an autoencoder consists of three main components: the encoder, the bottleneck (or latent space), and the decoder. The **encoder** function $h=f(x)$ compresses the input data  into a 'latent-space' representation $h$. This **latent-space** is a low-dimensional  space that captures the essential features  of the input data. Finally, the **decoder** function $r=g(h)$ reconstructs the input data from the compressed representation in the bottleneck.

<div style="
  background: white;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
">
  <img
    src="https://www.mathworks.com/discovery/autoencoder/_jcr_content/thumbnail.adapt.1200.medium.svg/1636390843825.svg"
    width="500"
    alt="autoencoder diagram"
  >
</div>


If an autoencoder succeeds in simply learning to set $g(f(x))=x$ everywhere, then it is not especially useful. Instead, autoencoders are designed to be unable to learn to copy perfectly. Usually they are restricted in ways that allow them to copy only approximately, and to copy only input that resembles the training data. Because the model is forced to prioritize which aspects of the input should be copied, it learns useful properties of the data.


## Resources
1. [Autoencoders - CH 14](https://www.deeplearningbook.org/contents/autoencoders.html)
2. [Autoencoders | Deep Learning Animated](https://www.youtube.com/watch?v=hZ4a4NgM3u0)
3. [Intro_To_AutoEncoders.ipynb](https://github.com/priyammaz/PyTorch-Adventures/blob/main/PyTorch%20for%20Generation/AutoEncoders/Intro%20to%20AutoEncoders/Intro_To_AutoEncoders.ipynb)
4. [Variational Autoencoders | Generative AI Animated](https://www.youtube.com/watch?v=qJeaCHQ1k2w)
5. [Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937)
6. [Reproducing Neural Discrete Representation Learning](https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/final_project.pdf)
7. [Vector Quantized Variational AutoEncoder (VQVAE) From Scratch](https://www.youtube.com/watch?v=1mi2MSvigcc) 🎥
8. [Vector_Quantized_Variational_AutoEncoders.ipynb](https://github.com/priyammaz/PyTorch-Adventures/blob/main/PyTorch%20for%20Generation/AutoEncoders/Intro%20to%20AutoEncoders/Vector_Quantized_Variational_AutoEncoders.ipynb)
9. [Vector-Quantized Variational Autoencoders (VQ-VAEs)](https://www.youtube.com/watch?v=yQvELPjmyn0) 🎥
10. [Residual Vector Quantization (RVQ) From Scratch](https://www.youtube.com/watch?v=ZnyfaQRQ8GI) 🎥
