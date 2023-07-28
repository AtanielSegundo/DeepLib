# DeepLib 1.0

**Uma library de machine learning leve e modular para Python 3, feita com o intuito de facilitar a implementação de redes neurais profundas.**

**Disclaimer: Este projeto foi desenvolvido por um acadêmico como forma de implementar e aplicar conceitos aprendidos sobre  machine learning e programação. Embora tenham sido empregados esforços para garantir a qualidade, o projeto pode conter limitações ou erros. Use por sua conta e risco.**

## Dependências:

- **numpy**
- **random**
- **copy**
- **collections**

## Classes e metodos disponiveis:

### [layers](https://github.com/seu_usuario/seu_repositorio/blob/master/DeepLib.py##L7C14-L7C14):

- **Layer**: Classe abstrata que representa uma camada (layer) em uma rede neural. Possui métodos para inicialização de pesos e vieses, propagação forward,retropropagação e metodos abstratos relativos a função de ativação e sua derivada.

- **Relu**: Classe que representa a função de ativação ReLU (Rectified Linear Unit). 

- **Sigmoid**: Classe que representa a função de ativação Sigmoid. 

- **Linear**: Classe que representa a função de ativação Linear (identidade). 

- **Softsign**: Classe que representa a função de ativação Softsign. 

- **Leaky_Relu**: Classe que representa a função de ativação Leaky ReLU. 

- **Tanh**: Classe que representa a função de ativação Tangente Hiperbólica (tanh).

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


 
