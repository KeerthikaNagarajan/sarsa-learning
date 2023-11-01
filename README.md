# SARSA Learning Algorithm
## AIM
Write the experiment AIM.

## PROBLEM STATEMENT
Explain the problem statement.

## SARSA LEARNING ALGORITHM
Include the steps involved in the SARSA Learning algorithm

## SARSA LEARNING FUNCTION
```python
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    # Write your code here
    select_action=lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
    alphas=decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
    epsilons=decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        next_state,reward,done,_=env.step(action)
        next_action=select_action(next_state,Q,epsilons[e])
        td_target=reward + gamma * Q[next_state][next_action]*(not done)
        td_error=td_target - Q[state][action]
        Q[state][action]=Q[state][action] + alphas[e] * td_error
        state, action=next_state,next_action
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s: {s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]    
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
Mention the optimal policy, optimal value function , success rate for the optimal policy.
### optimal policy
### optimal value function
### success rate for the optimal policy

### Plot comparing the state value functions of Monte Carlo method and SARSA learning.
#### State value functions of Monte Carlo method
<img width="651" alt="image" src="https://github.com/KeerthikaNagarajan/sarsa-learning/assets/93427089/2c7a4d94-cf18-4d40-b574-4ba505053c1f">

#### State value functions of SARSA learning
<img width="650" alt="image" src="https://github.com/KeerthikaNagarajan/sarsa-learning/assets/93427089/d4728587-5e1b-413f-a3ab-4b4ca3c24f4d">


## RESULT:

Write your result here
