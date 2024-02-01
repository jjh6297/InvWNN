

# Weight Nowcasting Network (WNN)

Code for ["Learning to Unlearn by Iterative Approximation of Past Weights"]


### Dependency


<!-- dependencies: -->
tensorflow >= 2.9.0 , 
tfimm >= 0.2.14 
<!-- | tensorflow | 2.3.0, 2.4.1 | <= 2.0 | -->


## Pre-trained Weights
Pre-trained weights of WNN are included.
'InvWNN_XXX.h5 ' in this repo are the pre-trained weights for each mathematical operation type (Conv, FC, Bias).


## Experiments

Unlearning Label-noised CIFAR10 from ResNet18:

```
python ./ResNet_CIFAR10_LabelNoise/InvWNN_Unlearning_ResNet_LabelNoise.py
```


Unlearning Label-noised CIFAR100 from PVTv2:

```
python ./PVTv2_CIFAR100_LabelNoise/InvWNN_Unlearning_PVTv2_LabelNoise.py
```

