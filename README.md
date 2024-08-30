# Football Bet Predictor

As name suggests, this project aims to predict odds of football matches and use these to generate profit by betting on potentionally profitable matches.

## Original work

This work is based on dataset from [this project](https://github.com/enricocattaneo/football-match-prediction). Dataset itself is not stored on GitHub due to it's size and can be downloaded from [Kaggle](https://www.kaggle.com/datasets/enricocattaneo/data-football-match-prediction). 

The author of the work aimed toward the same goal and did splendid job creating the dataset. Unfortunately though, in my opinion, used data rather poorly.

The first of all, one feature subset is composed entirely of betting odds from betting companies we are trying to beat. Not only betting odds were used as a feature to predict betting odds, 132 of 210 features were betting odds.

The second thing is, prediction accuracy is not the best indicator here. This particular classification task is not as straightforward as classifying images of dogs and cats. The beauty of football is that sometimes an underdog can beat favorit despite odds but cat is always cat and dog is a dog. On the other hand, literature so far fails to provide convincing measure to this task, at least to my best knowledge. I decided to judge the classificator in conjunction with betting strategy, not separately.

## Approach

Tried many different classificators such as kNN, SVM or random forests but none of them has performance even remotely comparable to neural network. NN is a simple 4 layer MLP in PyTorch. NNs with less layers were not reliable enough. Hidden layers have tens of neuron each as wide networks proved to be *too good*. Contrary to popular belief, output of the NN classifier is not probability of each class but something I've seen named confidence score. Wider NNs in this case tend to be very confident in their guesses. It means the confidence score of the most probable class is higher than real probability of the class which renders this approach useless. One solution is to calibrate output of the NN, the second one is to make NN *dumber*. Both solutions move output closer to real probability for each class. Calibrations creates transformation function from NN output to probability. It can be trained either on separate data or, if dataset is small, on the same data as NN. Smaller NN, on the other side, has to generalise more as it doesn't have enough space to memorise. While it sounds like overfitting, it is different concept. More on the topic of probability/confidence can be found via searching for *confidence calibration*.

Betting strategy is to bet 5% of available cash on any match that satisfies following condition

$$\frac{O_b}{O_p} = O_b P_p \gt 1 + \tau$$

where $O_b$ is odd from a bookie, $P_p$ is predicted probability of given result, $O_p$ is predicted odd equal to $1/P_p$ and $\tau$ is a margin.

## Results

Following graph shows cash over ~1200 matches.

![Alt text](/imgs/shuffle_rs27.png?raw=true)

The agent tends to be very trigger happy, betting on ~80% matches depending on chosen architecture. The most successful agent shown in the image was able to increase original amount of money by 125. The amount of course depends on architecture and also how data is shuffled. Games at the end of season are easier to predict so if there are more end-season games in the test batch, profit is higher. Honestly, it is the case here and image is an eye catcher :). Normal profit ranges from 5-25 which is still more than great. Keep in mind that if data is not shuffled, test data consists only of end-season game and profit can shoot up by order of a magnitude.