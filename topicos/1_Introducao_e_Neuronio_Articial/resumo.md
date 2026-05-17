Este tópico apresenta uma introdução à Inteligência Artificial e ao Aprendizado de Máquina, destacando a transição da IA clássica/simbólica, baseada em regras e lógica, para a IA moderna, fundamentada no aprendizado a partir de dados.

Inspirado nas ideias de Daniel Kahneman sobre pensamento rápido (intuitivo) e pensamento lento (deliberativo), o conteúdo aborda como o aprendizado de máquina surgiu como uma tentativa de aproximar os sistemas computacionais do comportamento humano, especialmente por meio de modelos probabilísticos e adaptativos. Também são apresentados os principais paradigmas do aprendizado de máquina: supervisionado, não supervisionado e por reforço.

Em seguida, é introduzido o perceptron, considerado um dos primeiros modelos de neurônio artificial. O perceptron consiste em um sistema que recebe entradas, aplica pesos a essas entradas e adiciona um bias, produzindo uma soma ponderada chamada potencial de ativação.

Como o potencial de ativação possui natureza linear, ele não é suficiente para representar problemas complexos do mundo real. Para resolver essa limitação, utiliza-se uma função de ativação, responsável por introduzir não linearidade ao modelo e permitir a representação de padrões mais complexos.

O treinamento de um perceptron consiste no ajuste dos pesos e do bias com base no erro da predição, utilizando funções de custo, como MSE para regressão e Entropia Cruzada para classificação, juntamente com algoritmos de otimização baseados em gradiente, especialmente a descida do gradiente.

O tópico também discute as principais funções de ativação utilizadas em redes neurais profundas, como Sigmoide, ReLU, Softmax e Tanh, destacando suas características, aplicações e limitações, incluindo problemas clássicos como vanishing gradient e dying ReLU.

Por fim, são abordadas a importância da taxa de aprendizado, responsável por controlar a velocidade e estabilidade do treinamento, e as principais estratégias de treinamento com batches, como Batch Gradient Descent, SGD e Mini-Batch Gradient Descent.