# CutOut (Pytorch)

CutOut is a regularization method that erases a part of the area in each selected image at random.

[Reference] Terrance DeVries, Graham W. Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout", arXiv preprint arXiv:1708.04552}, 2017

## Methodology
You can use the method simply by putting this class into your code.
```python
class CutOut:
    
    def __init__(self, ratio=.5):
        self.ratio = int(1/ratio)
           
    def __call__(self, inputs):

        active = int(np.random.randint(0, self.ratio, 1))
        
        
        if active == 0:
            _, w, h = inputs.size()
            min_len = min(w, h)
            w_c = int(np.random.randint(2, 8, 1))
            h_c = int(np.random.randint(2, 8, 1))
            w_size = int(min_len//w_c)
            h_size = int(min_len//h_c)
            th = max(w_size, h_size)
            idx = int(np.random.randint(0, min_len-th, 1))
            inputs[:,idx:idx+w_size,idx:idx+h_size] = 0
        
        return inputs
```
```python
transf = tr.Compose([tr.Resize(128), tr.ToTensor(), CutOut()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transf)
```
## CutOut Images
![cutout_cifar10](https://user-images.githubusercontent.com/52735725/115238312-d5eaf200-a11d-11eb-8ed2-87168cb15bb1.png)

