# hw2 Solution
## experiment 1
### CartPole
Run multiple experiments with the PG algorithm on the discrete CartPole-v0 environment.


Compare the learning curves (average eval return at each iteration) for the experiments 
run the following 6 commands respectively by adding 
configuration in hw2/cs285/scripts/run_hw2 from PyCharm ide

*Small* batch cases:

`--env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name q1_sb_no_rtg_dsa`

`--env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name q1_sb_rtg_dsa`

`--env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name q1_sb_rtg_na`

*Large* batch cases:

`--env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name q1_lb_no_rtg_dsa`

`--env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name q1_lb_rtg_dsa`

`--env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name q1_lb_rtg_na`
#### Create two graphs
- The first graph compare the learning curves (average eval return at each iteration for the experiments
prefixed with q1_sb_ of the small batch experiments), use 
  exponential weighted average method to smooth the curve

![](results/q1_sb.png)

- The second graph compare the learning curves for the experiments prefixed 
  with q1_lb.
  
![](results/q1_lb.png)

#### Answer the following questions
- Which value estimator has better performance without advantage-standardization: 
  the trajector centric one, or the one using reward-to-go?<br>
  - the reward-to-go method has better performance. From my perspective, Using this 
  method makes each gradient calculation focus only on the data after the current data, which reduces the variance <br>
  

- Did advantage standardization help?<br>
  - Yes. Normalization reduces the volatility of the advantage, thereby reducing the variance<br>
  

- Did the batch size make an impact?
  - Yes, larger batch size stabilize the result.

## experiment 2
