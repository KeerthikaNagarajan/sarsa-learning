# SARSA Learning Algorithm
## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method

## PROBLEM STATEMENT
The problem statement is a Five stage slippery walk where there are five stages excluding goal and hole.The problem is stochastic thus doesnt allow transition probability of 1 for each action it takes.It changes according to the state and policy.

#### State Space:
The states include two terminal states: 0-Hole[H] and 6-Goal[G].
It has five non terminal states includin starting state.

#### Action Space:
Left:0
Right:1

#### Transition probability:
The transition probabilities for the problem statement is:
50% - The agent moves in intended direction.
33.33% - The agent stays in the same state.
16.66% - The agent moves in orthogonal direction.

#### Reward:
To reach state 7 (Goal) : +1 otherwise : 0

## SARSA LEARNING ALGORITHM
1. Initialize Q-table with zeros, where each row represents a state, and each column represents an action.
2. Define an epsilon-greedy strategy to select actions: With probability epsilon, choose a random action, otherwise choose the action with the highest Q-value for the current state.
3. Set up learning rate (alpha) and exploration rate (epsilon) decay schedules for gradually reducing these parameters over episodes.
4. Loop through a fixed number of episodes (n_episodes), initializing the environment and setting the initial state and action.
5. In each episode, interact with the environment by taking actions, observing rewards, and transitioning to the next state until the episode terminates.
6. Update the Q-values using the SARSA update rule, incorporating the observed reward and the estimated future Q-value of the next state-action pair.
7. Track the Q-values, policy, and other information over episodes and return the final Q-values, state values (V), policy (pi), Q-value history (Q_track), and policy history (pi_track).

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
### Optimal Policy
<img width="587" alt="image" src="https://github.com/KeerthikaNagarajan/sarsa-learning/assets/93427089/80eac07c-62db-48e6-bc52-14fa509710f0">

### First Visit Monte Carlo Method:
<img width="527" alt="image" src="https://github.com/KeerthikaNagarajan/sarsa-learning/assets/93427089/6e59a2ef-27b9-49d0-bb7f-783b8a63d24e">

### SARSA Learning Algorithm:
<img width="538" alt="image" src="https://github.com/KeerthikaNagarajan/sarsa-learning/assets/93427089/daa75635-6a8e-43a8-84a7-da23b6a2891c">

### Plot comparing the state value functions of Monte Carlo method and SARSA learning:
#### State value functions of Monte Carlo method
<img width="648" alt="image" src="https://github.com/KeerthikaNagarajan/sarsa-learning/assets/93427089/b144affa-d0b7-4378-b908-2a13c49cbcb5">

#### State value functions of SARSA learning
<img width="650" alt="image" src="https://github.com/KeerthikaNagarajan/sarsa-learning/assets/93427089/ab73465e-9a76-43ad-98bb-7fb35341c2a0">

## RESULT:
Thus, the implementation of SARSA learning algorithm was implemented successfully.

