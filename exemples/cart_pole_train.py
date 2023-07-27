from DeepLib import *
import gym

#%% Inicializando o ambiente
env = gym.make("CartPole-v1")
stt_size =  env.observation_space.shape[0]
act_size = env.action_space.n
#%% Montando o agente

agente = DDQN()

agente.set_parameter(
                   state_size    = stt_size,
                   action_size   = act_size,
                   memory_size   = 2000,
                   alpha         = 0.005,
                   gamma         = 0.95,
                   policy        = agente.exp_epsilon_greedy,
                   epsilon       = 1.0,
                   epsilon_decay = 0.9989,
                   epsilon_min   = 0.01,
                   batch_size    = 16,
                   loss_function = agente.MSE_Loss
                   )
    
agente.add_layer(         
                layers.Relu(stt_size,32,"xavier"),
                layers.Relu(32,32,"xavier"),
                layers.Relu(32,16,"xavier"),
                layers.Leaky_Relu(16,2,"xavier")
                )

#%% Treinando o agente
EPISODES = 2000
for epoch in range(EPISODES):
    state,_ = env.reset()
    state = np.reshape(state, [1, stt_size])
    for time in range(800):
        action = agente.act(state)
        next_state , reward , end , _ , _ = env.step(action)
        
        x,x_dot,theta,theta_dot = next_state
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        
        next_state = np.reshape(next_state, [1, stt_size])
        agente.memorizar(state,action,reward,next_state,end)
        state = next_state
        if end:
            print(f"episode: {epoch}/{EPISODES}, score: {time}, e: {agente.epsilon}",end="\r")
            break
    agente.policy_update()
env.close()
total_reward = 9999
agente.save(path = "./models",name = f"cart_pole_{int(total_reward)}")

