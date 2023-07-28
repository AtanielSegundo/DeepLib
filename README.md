# DeepLib 1.0

**Uma library de machine learning leve e modular para Python 3, feita com o intuito de facilitar a implementação de redes neurais profundas.**

**Disclaimer: Este projeto foi desenvolvido por um acadêmico como forma de implementar e aplicar conceitos aprendidos sobre  machine learning e programação. Embora tenham sido empregados esforços para garantir a qualidade, o projeto pode conter limitações ou erros. Use por sua conta e risco.**

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
### DNN(Deep Neural Net):

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
     - Salva as camdas do modelo e um txt com os hiperparâmetros utilizados no subdiretorio o nome "name" no diretorio "path".  
   - **load**
     - Os argumentos que devem ser passados para o método são "path" e "load_cfg".
     - Por padrão,"load_cfg" = True.
     - Se load_cfg for True, o método também carrega os hiperparâmetros utilizados durante o treinamento.   
     - O método restaura a rede neural com os pesos, vieses e hiperparâmetros carregados para continuar o treinamento ou fazer previsões em novos dados.
    
**Exemplo de uso**:
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

### DDQN(Double Deep Quality Net):


## Sobre o Repositório

### [Exemplos](exemples)

- Pasta onde estão os scripts de exemplo de uso das classes e métodos disponíveis na biblioteca.

### [Modelos](models)

- Pasta onde estão disponíveis modelos já treinados utilizando a biblioteca.

### [DeepLib.py](DeepLib.py)

- Script Python da biblioteca com as classes e métodos relacionados a redes neurais profundas.

## Como Contribuir

Se você deseja contribuir para o desenvolvimento da DeepLib, siga estas etapas:

1. Faça um fork deste repositório.
2. Crie um branch para suas alterações: `git checkout -b minha-feature`
3. Faça suas modificações e commit: `git commit -m 'Adicionar nova feature'`
4. Envie para o branch principal do seu fork: `git push origin minha-feature`
5. Crie um novo Pull Request.

## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Contatos
Email: 10ataniel@gmail.com


 
