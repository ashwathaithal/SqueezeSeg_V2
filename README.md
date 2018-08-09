# SqueezeSeg_V2
1. Replace cross entropy loss with Focal loss, which punish more on hard-to-classify points. Useful in our case since we have large amount of background points.
2. Add batch normalizations to reduce overfitting

Update:
1. src/nets/squeezeseg.py: change all conv layers to conv_bn layers, put a prior distribution on the last conv.
2. src/nn_skeleton.py: change the loss function to focal loss.


Focal loss Reference: https://arxiv.org/pdf/1708.02002.pdf

Result:
Car iou: 0.7231, Car precision: 0.8102, Car recall: 0.8705

<p align="center">
    <img src="https://github.com/xuanyuzhou98/SqueezeSeg_V2/raw/master/readme/train.png" width="600" />
</p>
