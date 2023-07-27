from DeepLib import *
import gym

agente = DDQN()
agente.load("./models/cart_pole_9999")
agente.epsilon = 0
env = gym.make("CartPole-v1", render_mode = "human")
stt_size =  env.observation_space.shape[0]
act_size = env.action_space.n
done = False
state, _ = env.reset()
total_reward = 0
test = 1 
while not done and (test<10_000):    
    state = np.reshape(state, [1, stt_size])
    action = agente.act(state)
    next_state, reward, done, _, _ = env.step(action)
    total_reward += reward
    state = next_state
    env.render()
    test+=1

env.close()
print("Recompensa total = ",total_reward)