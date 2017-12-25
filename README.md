# Implementing the PowerSign and AddSign rule
The PowerSign and AddSign update rule are a gradient 
update rules found by Quoc V. Le's team and was 
reported in their paper "Neural Optimiser search with Reinforcment learning", which claimed that the optimiser found by their algorthim is better than adam and SGD for training neural networks. This repo tries to replicate their experiment for small CNNs in pytorch. I nickname my implementation of the PowerSign Rule NASoptimiser which stands for Neural Architectural Search optimiser. 

## AddSign 
![alt text](https://github.com/Neoanarika/Implementing-the-PowerSign-and-AddSign-rule/blob/master/img/addsign.png)

## PowerSign 
![alt text](https://github.com/Neoanarika/Implementing-the-PowerSign-and-AddSign-rule/blob/master/img/powersign.png)

# How to use 
```
python Addsign.py 
python Powersign.py
```
To use for your projects, copy the Addsign/Powersign file into ur project directory then import them respectively. 

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

Both PowerSign and AddSign have high training accuracy but much lower test accuracy, which suggest Overfitting is a common problem in update rules found by the Algorthim. 

By halfing the number of epochs we were able to reduce overfitting in PowerSign, this suggest early stopping is a good strategy to manage overfitting for PowerSign and possibly even for AddSign. 

We also observe that eventually PowerSign and AddSign eventually converged to the same value, I think this is because of the model we tested them on rather than properties of the optimiser themselves. 

## Conclusion 

We manage to get a similar graph that Quoc v. Le's team reported in their paper and provide a pytorch implementation of their optimiser for others to use in their experiments. 

### Original Results on small CNN
![alt text](https://github.com/Neoanarika/Implementing-the-PowerSign-and-AddSign-rule/blob/master/img/original.png)

### Our Results on small CNN
![alt text](https://github.com/Neoanarika/Implementing-the-PowerSign-and-AddSign-rule/blob/master/img/300%20epcohs%20all.png)

## Adavantages to using Addsign and Powersign over adam

Addsign and Powersign only has a single memory buffer comapred to adam double memory buffer, hence it is more memory effective although it can yield similar or better result than adam. 

# Dependecies 
```
1. Python 3
2. Pytorch 
```

Make sure that the Pytorch can support GPU. 

Working on a tensorflow implementation soon

# References 
1. Bello, Irwan, Barret Zoph, Vijay Vasudevan, and Quoc V. Le. "Neural optimizer search with reinforcement learning." arXiv preprint arXiv:1709.07417 (2017).
