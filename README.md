# DeepLib 1.0

**Uma library de machine learning leve e modular para Python 3, feita com o intuito de facilitar a implementação de redes neurais profundas.**

**Disclaimer: Este projeto foi desenvolvido por um acadêmico como forma de implementar e aplicar conceitos aprendidos sobre  machine learning e programação. Embora tenham sido empregados esforços para garantir a qualidade, o projeto pode conter limitações ou erros. Use por sua conta e risco.**

## Dependências:

- **numpy**
- **random**
- **copy**
- **deque**

## Classes e metodos disponiveis:

### layers

- **Layer**: Classe abstrata que representa uma camada (layer) em uma rede neural. Possui métodos para inicialização de pesos e vieses, propagação forward e retropropagação não paramétrica.

- **Relu**: Classe que representa a função de ativação ReLU (Rectified Linear Unit). Possui métodos para a função de ativação ReLU e sua derivada.

- **Sigmoid**: Classe que representa a função de ativação Sigmoid. Possui métodos para a função de ativação Sigmoid e sua derivada.

- **Linear**: Classe que representa a função de ativação Linear (identidade). Possui métodos para a função de ativação Linear e sua derivada.

- **Softsign**: Classe que representa a função de ativação Softsign. Possui métodos para a função de ativação Softsign e sua derivada.

- **Leaky_Relu**: Classe que representa a função de ativação Leaky ReLU. Possui métodos para a função de ativação Leaky ReLU e sua derivada.

- **Tanh**: Classe que representa a função de ativação Tangente Hiperbólica (tanh). Possui métodos para a função de ativação tanh e sua derivada.

```python
from DeepLib import Layer

input_size = 10
output_size = 5
initializer = "he"  # ou "xavier" ou "std"
act_func = lambda x: x  # Função de ativação linear
derivative = lambda x: 1  # Derivada da função de ativação linear

layer = Layer(input_size, output_size, initializer, act_func, derivative)

# Faça a propagação forward através da camada passando um input (vetor) como argumento
input_data = np.random.rand(1, input_size)  # Exemplo de um vetor de entrada de tamanho 1x10
output_data = layer(input_data)

print("Saída da camada:")
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


 
