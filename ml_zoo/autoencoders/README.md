# Autoencoders
Anautoencoderis a neural network that is trained to attempt to copy its input to its output. Internally, it has a hidden layer $h$ that describes acodeused torepresent the input. The network may be viewed as consisting of two parts: an **encoder** function $h=f(x)$ and a **decoder** that produces a reconstruction $r=g(h)$.

<p align="center">
  <img src="https://www.mdpi.com/IoT/IoT-04-00016/article_deploy/html/images/IoT-04-00016-g001.png" alt="alt" width="500">
</p>

If an autoencoder succeeds in simplylearning to set $g(f(x))=x$ everywhere, then it is not especially useful. Instead,autoencoders are designed to be unable to learn to copy perfectly. Usually they arerestricted in ways that allow them to copy only approximately, and to copy onlyinput that resembles the training data. Because the model is forced to prioritizewhich aspects of the input should be copied, it often learns useful properties of the data.


## References
1. https://www.deeplearningbook.org/contents/autoencoders.html
2. [Reproducing Neural Discrete Representation Learning](https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/final_project.pdf)
3. [Residual Vector Quantization (RVQ) From Scratch](https://www.youtube.com/watch?v=ZnyfaQRQ8GI)
4. [Vector_Quantized_Variational_AutoEncoders.ipynb](https://github.com/priyammaz/PyTorch-Adventures/blob/main/PyTorch%20for%20Generation/AutoEncoders/Intro%20to%20AutoEncoders/Vector_Quantized_Variational_AutoEncoders.ipynb)