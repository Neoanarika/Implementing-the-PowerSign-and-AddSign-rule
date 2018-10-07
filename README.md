# Implementing the PowerSign and AddSign rule
The PowerSign and AddSign update rules are gradient 
update rules found by Quoc V. Le's team. It was 
reported in their paper "Neural Optimiser search with Reinforcment learning", which claimed that the optimiser found by their algorithm is better than adam and SGD for training neural networks. This repo attempts to replicate their experiment for small CNNs in pytorch. 

Update : Tensorflow implementation is added as of 25 Feb 2018 for tensorflow 1.5.

## AddSign 
![alt text](https://github.com/Neoanarika/Implementing-the-PowerSign-and-AddSign-rule/blob/master/img/addsign.png)

## PowerSign 
![alt text](https://github.com/Neoanarika/Implementing-the-PowerSign-and-AddSign-rule/blob/master/img/powersign.png)

# How to use 
```
python Addsign.py 
python Powersign.py
```
To use in your projects, copy the Addsign/Powersign file into your project directory and import them respectively. 

## Addsign
```
from Addsign import Addsign
optimizer = AddSign(cnn.parameters(), lr=0.01)
```

## Powersign
```
from Powersign import Powersign
optimizer = Powersign(cnn.parameters(), lr=0.01)
```

# Results 

## 160 Epochs 
![alt text](https://github.com/Neoanarika/Implementing-the-PowerSign-and-AddSign-rule/blob/master/img/160%20epochs.png)


| Optimiser     | Colour        |
| ------------- | ------------- |
| SGD           | Sky Blue      |
| Adam          | Red           |
| PowerSign     | Dark Blue     |
| AddSign       | Orange        |

## 300 Epochs
![alt text](https://github.com/Neoanarika/Implementing-the-PowerSign-and-AddSign-rule/blob/master/img/300%20epochs.png)
![alt text](https://github.com/Neoanarika/Implementing-the-PowerSign-and-AddSign-rule/blob/master/img/300%20epcohs%20time.png)

Both PowerSign and AddSign have high training accuracy but much lower test accuracy, suggesting that overfitting is a common problem in update rules found by the algorthim. 

By halving the number of epochs, we were able to reduce overfitting in PowerSign, suggesting that early stopping is a good strategy to manage overfitting for PowerSign and possibly even for AddSign. 

We also observed that PowerSign and AddSign eventually converges to the same value, which I think is due to the model we tested them on rather than the properties of the optimiser themselves.

## Conclusion 

We managed to get a similar graph that Quoc v. Le's team reported in their paper and provided a pytorch implementation of their optimiser for others to use in their experiments. 

### Original Results on small CNN
![alt text](https://github.com/Neoanarika/Implementing-the-PowerSign-and-AddSign-rule/blob/master/img/original.png)

The original results initally called Powersign as Optimiser_1. 

### Our Results on small CNN
![alt text](https://github.com/Neoanarika/Implementing-the-PowerSign-and-AddSign-rule/blob/master/img/300%20epcohs%20all.png)

## Adavantages to using Addsign and Powersign over adam

Addsign and Powersign has only a single memory buffer compared to adam's double memory buffer, hence it is more memory effective although it can yield similar, if not better results than adam. 

# Dependencies 
```
1. Python 3
2. Pytorch 
3. tensorboardX # To allow pytorch to use tensorboard
4. tensorflow 1.5
```

Make sure that Pytorch can support GPU. 

Working on a tensorflow implementation soon.

# References 
1. Bello, Irwan, Barret Zoph, Vijay Vasudevan, and Quoc V. Le. "Neural optimizer search with reinforcement learning." arXiv preprint arXiv:1709.07417 (2017).
