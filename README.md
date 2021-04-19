# CutOut

CutOut is a regularization method that erases a part of the area in each selected image at random.

[Reference] Terrance DeVries, Graham W. Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout", arXiv preprint arXiv:1708.04552}, 2017

# Methodology
Add this class to your code.
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
