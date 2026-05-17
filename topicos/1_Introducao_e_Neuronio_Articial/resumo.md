Este tópico de conteúdo apresenta uma introdução à Inteligência Artificial e ao Aprendizado de Máquina, destacando a transição da IA clássica/simbólica — baseada em regras e lógica — para a IA moderna, baseada em aprendizado a partir de dados. 

Inspirado nas ideias de Daniel Kahneman sobre pensamento rápido (intuitivo) e pensamento lento (deliberativo), aprendemos sobre como o aprendizado de máquina surgiu como uma tentativa de aproximar os sistemas computacionais do comportamento humano, especialmente através de modelos probabilísticos e adaptativos. Também foram apresentados os principais paradigmas do aprendizado de máquina: supervisionado, não supervisionado e por reforço.

Em seguida, conhecemos o perceptron, considerado um dos primeiros modelos de neurônio artificial já criados. O modelo de neurônio artificial designado pelo perceptron consiste em um sistema que recebe entradas, as quais aplica pesos por meio de multiplicação e por fim adiciona um bias para viés. O resultado dessa espécie de soma ponderada é chamado de potencial de ativação.

O potencial de ativação, devido a sua própria estrutura, é uma reta, que não consegue descrever problemas complexos do mundo real. Portanto, precisa-se adicionar não-linearidade, e para isso, o potencial de ativação é passado pela chamada função de ativação que permite modular essa saída para problemas complexos. 

O treinamento de um perceptron consiste em ajustar pesos e bias com base no erro da predição, utilizando funções de custo (como MSE para problemas de regressão e Entropia Cruzada para problemas de classificação) e algoritmos de otimização baseados em gradiente, especialmente a descida do gradiente. 

Ainda nesse mesmo tópico, são discutidas as principais funções de ativação utilizadas em redes neurais profundas, como Sigmoide, ReLU, Softmax e Tanh, destacando suas características, aplicações e limitações — incluindo problemas clássicos como vanishing gradient e dying ReLU. 

Discutimos também sobre a importância da taxa de aprendizado, que pode influenciar na resposta do modelo, além das diferentes estratégias de treinamento com batches, como Batch Gradient Descent, SGD e Mini-Batch Gradient Descent. 