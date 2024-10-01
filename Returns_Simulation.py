import numpy as np
import matplotlib.pyplot as plt


class SB_Path:
    """
    Cette classe construit m mouvements browniens standards, sur N pas
    """
    def __init__(self, m, N):
        self.m = m
        self.N = N
        self.dt = 1.0 / N  
        self.path = self.generate_brownian_path()

    def generate_brownian_path(self):
        """
        Ici on génère une matrice de m ligne contenant sur chaque ligne une trajectoire brownienne à N pas
        """
        dW = np.sqrt(self.dt) * np.random.randn(self.m, self.N)
        W = np.zeros((self.m, self.N + 1))
        W[:, 1:] = np.cumsum(dW, axis=1)
        return W
    
    def get_path(self):
        return self.path

    def plot_paths(self):
        """
        Pour tracer les trajectoires construites
        """
        t = np.linspace(0, 1, self.N + 1)  
        for i in range(self.m):
            plt.plot(t, self.path[i], label=f'Chemin {i + 1}')
        plt.title(f'{self.m} Standard brownien motions')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    


class GBM_correlated:
    def __init__(self, SIGMA, m, N):
        if not isinstance(SIGMA, np.ndarray) or SIGMA.ndim != 2:
            raise ValueError("SIGMA must be a matrix (2-dimensional np.ndarray).")
        
        if SIGMA.shape[0] != m or SIGMA.shape[1] != m:
            raise ValueError(f"SIGMA must be a square matrix of size {m}x{m}.")

        self.SIGMA = SIGMA
        self.m = m
        self.N = N

    def Cholesky(self):
        return np.linalg.cholesky(self.SIGMA)
    

    def generate_GBM(self):
        L = self.Cholesky()
        standard_BM = SB_Path(self.m, self.N)
        B = L @ standard_BM.get_path()
        return B
    
    def plot_correlated_paths(self):
        """
        Trace les trajectoires browniennes corrélées
        """
        correlated_BM = self.generate_GBM()
        
        t = np.linspace(0, 1, self.N + 1)  
        
        for i in range(self.m):
            plt.plot(t, correlated_BM[i], label=f'Corr. Path {i + 1}')
        
        plt.title(f'{self.m} Correlated Brownian Motions')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()


class Price:
    def __init__(self, S0, sigma, mu, SIGMA, m, N):
        self.S0 = S0
        self.sigma = sigma
        self.mu = mu
        self.SIGMA = SIGMA
        self.m = m
        self.N = N
    
    
    def simulate_prices(self):
        GBM = GBM_correlated(self.SIGMA, self.m, self.N)
        B = GBM.generate_GBM()
        
        prices = np.zeros((self.m, self.N + 1))
        
        prices[:, 0] = self.S0
        
        dt = 1 / self.N  
        for i in range(1, self.N + 1):
            prices[:, i] = self.S0 * np.exp(
                (self.mu - 0.5 * self.sigma**2) * (i * dt) + self.sigma * B[:, i]
            )
        
        return prices
    
    def plot_prices(self):
        """
        Trace l'évolution des prix pour chaque actif
        """
        prices = self.simulate_prices()
        t = np.linspace(0, 1, self.N + 1)  # Échelle de temps de 0 à 1
        
        for i in range(self.m):
            plt.plot(t, prices[i], label=f'Actif {i + 1}')
        
        plt.title(f'Évolution des prix de {self.m} actifs')
        plt.xlabel('Temps')
        plt.ylabel('Prix')
        plt.legend()
        plt.show()




















S0 = np.array([100, 105, 110])  # Prix initiaux
sigma = np.array([0.2, 0.25, 0.15])  # Volatilités
mu = np.array([0.05, 0.03, 0.04])  # Taux de croissance
SIGMA = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])  # Matrice de corrélation
m = 3  # Nombre d'actifs
N = 100  # Nombre de pas

# Créer un objet Price et tracer les prix simulés
price_simulator = Price(S0, sigma, mu, SIGMA, m, N)
price_simulator.plot_prices()  # Tracer les prix

