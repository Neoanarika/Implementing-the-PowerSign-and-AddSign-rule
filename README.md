# NASoptimser

Implementing the update learning rule found in Neural Optimizer search with Reinforcement learning. 

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
| NASOptimiser  | <b>75.0%</b>  |

## 300 Epochs
| Optimiser     | Test Accuracy |
| ------------- | ------------- |
| SGD           | 60.0%         |
| Adam          | <b>62.0%</b>  |
| NASOptimiser  | 50.0%         | 

# Dependecies 
```
1. Python 3
2. Pytorch 
```

Make sure that the Pytorch can support GPU. 

Working on a tensorflow implementation soon
