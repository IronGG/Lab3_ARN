## Introduction:

This is our practical work on "Learning with Artificial Neural Networks" under the supervision of Professor Andres Perez-Uribe, with assistance from Shabnam Ataee and Simon Walther. This session focuses on utilizing Multilayer Perceptrons (MLPs) with Keras to classify mice's sleep stages. The primary objectives are to apply acquired knowledge to real data.

During this practical work, participants engaged in the classification of mice's sleep stages using neural networks. EEG data from multiple mice will be employed for both training and testing purposes. Through prescribed preprocessing steps, model training, validation procedures, and performance assessments, we will gain practical insights into MLPs' efficacy in classification tasks.

## First experiment

### Our implementation

TODO : preprocessing -> why we did like that ? (fit_transform and transform)

First of all, we imported the data and structured it in a way that we could handle. We took the 25 first features and took the first column as the target (the column with the rat's state).

Then we implemented a basic program that created a MLP with keras. After that, the first thing we tried was changing the amount of layers and Perceptrons per layer. However, we didn't see a lot of difference at first. It looked like the problem we tried to handle was linearly separable or close to that since we reached F1_scores and accuracys around ~85% really fast. Then we tried adding neurons, incrasing epochs and changing optimizer functions. That made us reach around 90% micro f1_score and accuracy.

Optimisation functions we sticked to :

- TODO: *insert function name* 
- SGD 
- SGD with momentum.

To this program we quickly added the 3-fold that was required. 

The confusion matrix showed us interesting things :

TODO : *Insert confusion matrix*

### Problems found
Since it was the first time having to implement a MLP a lot of time was lost on understanding the libraries (Keras and matplotlib). 
Furthermore we wanted to try different optimizers and had to "custimize" them ourselves which took some time. 
Matplotlib was also complicated to handle at times and having it display the informations we wanted.


## Second experiment

### Our implementation

Compared to the last experiment since we had 1 more class it was likely that we needed more perceptrons 


### Problems found
