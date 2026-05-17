"""Implementacao de um neuronio artificial para regressao linear."""

import numpy as np

class ArtificialNeuron:
    """Neuronio simples para regressao linear usando gradiente descendente."""
    
    def __init__(self, alpha=0.01, b=None, w=None):
        """
        Inicializa o neuronio.
        
        Args:
            alpha: Taxa de aprendizado
            b: Bias inicial (intercepto)
            w: Peso inicial (inclinacao)
        """
        self.alpha = alpha
        self.b = b
        self.w = w
        self.b_history = []
        self.w_history = []

    def predict(self, x):
        """Faz a predicao para um valor x."""
        return self.w * x + self.b

    def compute_mse(self, data_x, data_y):
        """Calcula o Erro Quadratico Medio (MSE)."""
        predictions = self.predict(data_x)
        return ((predictions - data_y) ** 2).mean()

    def step_gradient(self, data_x, data_y):
        """Executa um passo do gradiente descendente."""
        predictions = self.predict(data_x)
        n = len(data_x)

        dw = (2/n) * ((predictions - data_y) * data_x).sum()
        db = (2/n) * (predictions - data_y).sum()

        self.w -= self.alpha * dw
        self.b -= self.alpha * db

    def fit(self, data_x, data_y, num_epochs=100, initial_b=0, initial_w=0):
        """
        Treina o neuronio usando gradiente descendente.
        
        Returns:
            Tupla (b_history, w_history) com o historico dos parametros
        """
        self.b = initial_b
        self.w = initial_w

        self.b_history = [self.b]
        self.w_history = [self.w]

        for _ in range(num_epochs):
            self.step_gradient(data_x, data_y)
            self.b_history.append(self.b)
            self.w_history.append(self.w)

        return self.b_history, self.w_history