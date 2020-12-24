



# DCGAN pytorch CIFAR10

Implement of DCGAN pytorch using CIFAR10  
[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, ICLR 2016](https://arxiv.org/abs/1511.06434)
<br>




![dcgan1](https://user-images.githubusercontent.com/37301677/79193816-58cfd180-7e66-11ea-8573-f8ffecd03627.png)



<h2>Train</h2>


```
python dcgan.py --dataroot [DATAROOT] --dataset [CIFAR] --model [DCGAN]
```



<h2>Configures</h2>



```
  model: dcgan
  is_train: True
  dataroot: dataset/cifar
  dataset: cifar
  download: True
  epochs: 25
  batch_size: 128
  image_size: 64
  nc: 3
  nz: 100
  ngf: 64
  ndf: 64
  learning_rate: 0.0002
  beta1: 0.5
  ngpu: 1
  cuda: True
  load_D: False
  load_G: False
  workers: 2
  generator_iters: 10000
  gpuids: [0]
```





<h2>Results</h2>


- generated images

![dcgan-cifar](https://user-images.githubusercontent.com/37301677/79194799-2921c900-7e68-11ea-8ced-49452a09b616.gif)


- logs

![dcgan_losses](https://user-images.githubusercontent.com/37301677/79195372-1f4c9580-7e69-11ea-8b8a-4cbe83029f32.png)


<h2>References</h2>


- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- https://github.com/pytorch/examples/tree/master/dcgan
- https://github.com/Zeleni9/pytorch-wgan
- https://github.com/sbarratt/inception-score-pytorch
- https://github.com/mseitzer/pytorch-fid

