from Environment import Factory
import numpy as np
import pandas as pd

performance = np.ones((30,6))
for n in range(30):
    for j in range(6):
        action = j
        env = Factory(n)
        state = env.reset()
        while True:
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                performance[n,j] = info
                break

p_df = pd.DataFrame(performance)
p_df.to_csv('single_dispatching_rule_tardy.csv', index=False)