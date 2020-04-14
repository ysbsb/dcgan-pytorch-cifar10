# DCGAN pytorch CIFAR10

![dcgan1](https://user-images.githubusercontent.com/37301677/79193816-58cfd180-7e66-11ea-8573-f8ffecd03627.png)

<br>

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



<br>

<h2>Train</h2>

```shell
python main.py --dataroot [DATAROOT] --dataset [CIFAR] --model [DCGAN]
```



<br>

<h2>References</h2>

- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- https://github.com/pytorch/examples/tree/master/dcgan
- https://github.com/Zeleni9/pytorch-wgan
- https://github.com/sbarratt/inception-score-pytorch
- https://github.com/mseitzer/pytorch-fid

<br>