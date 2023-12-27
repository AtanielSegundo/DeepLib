# DeepLib 1.0

**Disclaimer: Use por sua conta e risco.**

## Dependências:

- **numpy**
- **random**
- **copy**
- **collections**

## Classes e metodos disponiveis:

### layers:

- **Layer**: Classe abstrata que representa uma camada (layer) em uma rede neural. Possui métodos para inicialização de pesos e vieses, propagação forward,retropropagação e metodos abstratos relativos a função de ativação e sua derivada.

- **Relu**: Classe que representa a função de ativação ReLU (Rectified Linear Unit). 

- **Sigmoid**: Classe que representa a função de ativação Sigmoid. 

- **Linear**: Classe que representa a função de ativação Linear (identidade). 

- **Softsign**: Classe que representa a função de ativação Softsign. 

- **Leaky_Relu**: Classe que representa a função de ativação Leaky ReLU. 

- **Tanh**: Classe que representa a função de ativação Tangente Hiperbólica (tanh).

**Exemplo de uso**:
```python
import numpy as np
from DeepLib import layers

input_data = np.random.rand(1,10)  # Exemplo de um vetor de entrada de tamanho 1x10
initializer = "he"  # ou "xavier" ou "std"

layer1 = layers.Relu(input_size=10,output_size=16,init=initializer)
layer2 = layers.Tanh(input_size=10,output_size=32,init=initializer)

print("vetor de entrada")
print(input_data)
output_data = layer1(input_data)
print("Saída da camada 1:")
print(output_data)
output_data = layer2(input_data)
print("Saída da camada 2:")
print(output_data)
```
### DNN(Deep Neural Network):

**A classe DNN permite criar redes neurais profundas.**

- **Metodos:**
   - **_call_**
     - Metodo responsavel por fazer a propagação da entrada pela rede
     - Deve receber um vetor de entrada igual ao input size da primeira camada.
       
   - **fit**
     - Método que realiza o treinamento da rede neural utilizando o algoritmo de retropropagação por minibatch com otimização Adam.
     - Os argumentos para a função são Xtrain,Ytrain e epochs.
       - Xtrain: Vetor contendo os dados de entrada de treinamento, onde cada linha representa uma amostra.
       - Ytrain: Vetor contendo os rótulos (labels) correspondentes às amostras de treinamento.
       - epochs: Número de épocas (iterações completas pelo conjunto de treinamento) para treinar a rede (padrão: 1).
   - **set_parameter**
     - Metodo para configurar os hiperparâmetros da rede antes de seu treinamento.
     - Hiperparâmetros suportados:
       - alpha:
         - Controla a taxa que os pesos da rede neural são atualizados durante o treinamento. 
       - batch_size:
         - Determina quantos exemplos de treinamento são usados para atualizar os pesos da rede de uma vez na retropropagação por minibatch. 
       - delta:
         - O valor utilizado para calcular a perda Huber controla a sensibilidade do modelo a outliers.
       - loss_function:
         - Determina a função de perca usada pelo modelo, por padrão utiliza a função erro quadratico medio.
   - **save**
     - Os argumentos que devem ser passados para o metodo são "path" e "name".
     - Por padrão, path = "./models" e name = "no_named".
     - Salva no subdiretorio com o nome "name" no diretorio "path" as camdas do modelo e um txt com os hiperparâmetros utilizados.  
   - **load**
     - Os argumentos que devem ser passados para o método são "path" e "load_cfg".
     - Por padrão,"load_cfg" = True.
     - Se load_cfg for True, o método também carrega os hiperparâmetros utilizados durante o treinamento.   
     - O método restaura a rede neural com os pesos, vieses e hiperparâmetros carregados para continuar o treinamento ou fazer previsões em novos dados.
    
**Exemplo de uso**:

Criando uma rede neural que estima a função seno.
```python
from sklearn.model_selection import train_test_split as ts
from DeepLib import *

def test(Xtest,Ytest,precision=5e-3):
    Acuracy = 0
    Total = len(Xtest)
    for x,y in zip(Xtest,Ytest):
        pred = neural_net(x)
        if abs(float((pred))-y) <= precision:
            Acuracy += 1            
    Acuracy = Acuracy / Total
    return Acuracy

# Inicializando e configurando o modelo
neural_net = DNN()
neural_net.set_parameter(alpha=0.005,batch_size=16,loss_function=neural_net.Huber_Loss,delta=1.0)
init = "xavier"
neural_net.add_layer(layers.Relu(1,32,init),layers.Tanh(32,32,init),layers.Linear(32,1,init))

# Organizando os dados para o treinamento
X = np.linspace(0,2*np.pi,10_000)
Y = np.sin(X)
Xtrain,Xtest,Ytrain,Ytest = ts(X,Y,test_size=0.3,random_state=66)
train_zip = list(zip(Xtrain,Ytrain))

#Treinando o modelo usando minibatchs
for epoch in (epochs:=range(50)):    
    acuracy = test(Xtest,Ytest)
    print(f"Epoch = {epoch}, Acuracy = {acuracy}",end="\r")
    np.random.shuffle(train_zip)
    for i in range(0,len(train_zip),neural_net.batch_size):
        batch = train_zip[i:i+neural_net.batch_size]
        Xbatch, Ybatch = zip(*batch)
        Xbatch = np.array(Xbatch)
        Ybatch = np.array(Ybatch)
        neural_net.fit(Xbatch,Ybatch,epochs = 1)

#Salvando o modelo treinado
neural_net.save(path = "./models",name = f"sin_model_{acuracy}")

```

O modelo treinado com estes hiperparâmetros deve aproximar a senoide com a seguinte acuracia:

<p float="left">

<img src="https://github.com/AtanielSegundo/DeepLib/blob/master/Figure_sin.png" width="auto" />

</p>

Imagem gerada através do script [Sin Solution](https://github.com/AtanielSegundo/DeepLib/blob/master/exemples/sin_solution.py).


### DDQN(Double Deep Q Network):
**Classe que permite criar um agente capaz de estimar a função Q de um ambiente através da técnica DDQN.Nesta implementação de DDQN a rede principal e a rede alvo são amalgamadas.**

**Está classe herda todos os metodos da classe DNN.**

- **Metodos:**

  - **set_parameter**
    - Novos hiperparâmetros são suportados para a classe DDQN.
    - policy:
      - Politica de Exploraçaõa x Explotação que o agente deve seguir.
      - Por padrão a politica é a epsilon-greedy linear.
      - Politicas disponiveis:
        - linear_epsilon_greedy
          - o agente explore o ambiente aleatoriamente com probabilidade epsilon (ε) e escolha a ação que parece ser a melhor com probabilidade (1 - ε).
          - indicada para ambientes onde ações individuais podem ser claramente mapeadas e enumeradas (jogos de tabuleiro, jogos de cartas, labirintos com ações de movimento, etc...).
          - necessita de três hiperparâmetros auxiliares,epsilon,epsilon_min e epsilon_decay.
        - exp_epsilon_greedy
          -   uma variação da política linear epsilon-greedy. Nesta estratégia, a probabilidade de exploração (ε) é multiplicada por um fator de decaimento exponencial a cada atualização, ao invés de decair linearmente.
          -   a taxa de redução é mais gradual à medida que o agente ganha experiência.
          -   indicada para ambientes onde as ações possuem um espaço contínuo, como controlar a velocidade de um carro ou determinar o ângulo que um volante deve virar.
          -   necessita de três hiperparâmetros auxiliares,epsilon,epsilon_min e epsilon_decay.
        - boltzman/softmax
          - a política Boltzmann atribui probabilidades a cada ação com base em suas estimativas de recompensa (Q-values) e uma temperatura (τ).
          - uma temperatura alta (τ > 1) leva a uma distribuição de probabilidade mais uniforme
          - uma temperatura baixa (τ < 1) torna a política mais determinística, favorecendo ações com recompensas Q mais altas com maior probabilidade.
          - necessita do hiperparâmetro auxiliar temp.    
    - gamma:
       - Determina o desconto aplicado nas recompensas futuras.
       - É importante ajustar este hiperparâmetro para exista um equilíbrio entre recompensas imediatas e recompensas futuras.
       - Valores de gamma próximos a 0 dão pouco valor às recompensas futuras.
       - Valores próximos a 1 atribuem maior importância às recompensas futuras.
    - memory_size:
       - Um inteiro que representa o tamanho da memoria do agente.
       - A memoria é uma fila de acesso duplo que é usada para atualizar os pessos e vieses do agente.
       
  -   **act**
      - Metodo que recebe o estado do agente no ambiente e retorna o vetor com a estimativa da função Q para cada ação.
      - Necessario que o vetor estado tenha dimensões [1,S], sendo S a dimensão do input_size da primeira camada da DDQN.
  -   **policy_update**
      - Metodo que atualiza os hiperparâmetros auxiliares da politica do agente.
      - Utilizado geralmente no final do laço de um episodio de treinamento do agente.
  - **memorizar**
     - Recebe como argumento a tupla (estado,ação,recompensa,proximo_estado,flag de fim).
     - Necessario que o vetor estado e proximo_estado tenham dimensões [1,S], sendo S a dimensão do "input_size" da primeira camada da DDQN.
     - Metodo que memoriza a tupla memoria obtida após uma interação do agente com ambiente.
     - Após terem sidos memorizados um numero de exemplos iguais o batch_size, o agente retropropaga para suas camadas os gradientes.
     - Quando um numero de exemplos iguais a "memory_size" é adicionado as memorias a rede alvo é atualizada com os pesos e vieses da rede principal.

**Exemplo de uso**:

Criando um agente capaz de resolver o ambiente [Pêndulo de Carrinho](https://www.gymlibrary.dev/environments/classic_control/cart_pole/).

<center>
    <p float="left">
        <img src="https://www.gymlibrary.dev/_images/cart_pole.gif" width="auto" />
    </p>
</center>

```python
from DeepLib import *
import gym

#Inicializando o ambiente
env = gym.make("CartPole-v1")
stt_size =  env.observation_space.shape[0]
act_size = env.action_space.n
#%% Montando o agente

#Inicializando o agente
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

#Treinando o agente
for epoch in range(EPISODES:=2000):
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
agente.save(path = "./models",name = "cart_pole_solution")
```
Após o treinamento o modelo é capaz de estabilizar o pêndulo.

<p float="left">

<img src="https://github.com/AtanielSegundo/DeepLib/blob/master/cartpole_solution.png" width="auto" />

</p>

A simulação do modelo treinado pode ser feita utilizando o script [CartPole Solution](https://github.com/AtanielSegundo/DeepLib/blob/master/exemples/cart_pole_solution.py).

## Sobre o Repositório

### [Exemplos](exemples)

- Pasta onde estão os scripts de exemplo de uso das classes e métodos disponíveis na biblioteca.

### [Modelos](models)

- Pasta onde estão disponíveis modelos já treinados utilizando a biblioteca.

### [DeepLib.py](DeepLib.py)

- Script Python da biblioteca com as classes e métodos relacionados a redes neurais profundas.
- Para que você possa utilizar de forma adequada a DeepLib é necessário que ela esteja no mesmo diretorio dos seus scripts.

## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.


**Feedback é bem vindo.**



 
