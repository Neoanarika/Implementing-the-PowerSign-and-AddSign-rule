# Implementing the PowerSign and AddSign rule
The PowerSign and AddSign update rule are a gradient 
update rules found by Quoc V. Le's team and was 
reported in their paper "Neural Optimiser search with Reinforcment learning", which claimed that the optimiser found by their algorthim is better than adam and SGD for training neural networks. This repo tries to replicate their experiment for small CNNs in pytorch. I nickname my implementation of the PowerSign Rule NASoptimiser which stands for Neural Architectural Search optimiser. 

# How to use 
```
python CNN_with_NASoptim.py 
```

# Perliminary Results 

## 160 Epochs 
| Optimiser     | Test Accuracy |
| ------------- | ------------- |
| SGD           | 37.0%         |
| Adam          | 62.0%         |
| PowerSign     | <b>75.0%</b>  |
| AddSign       | 50.0%         |

## 300 Epochs
| Optimiser     | Test Accuracy |
| ------------- | ------------- |
| SGD           | 60.0%         |
| Adam          | <b>62.0%</b>  |
| PowerSign     | 50.0%         | 

This suggest the optimiser quickly overfits 
the CNN as early stopping improves the perofrmace 
of the model. 

The next step will be to visualise the learning on
tensorboard while running 500 epochs. 

# Dependecies 
```
1. Python 3
2. Pytorch 
```

Make sure that the Pytorch can support GPU. 

Working on a tensorflow implementation soon
