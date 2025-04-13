
from utils import vectors_to_image,img_to_vectors,plot_decision_regions
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np

def square_euclid(x, y):
    return np.sum((x - y)**2, axis=1)

def mse(img1, img2):
    return np.sum(square_euclid(img1, img2))/img1.size

class Gn:
    def __init__(self, n_prototypes=10, eta=0.1, n_epochs=10):
        self.k=n_prototypes
        self.eta = eta
        self.n_epochs = n_epochs

    def init_prototypes(self, X):
        self.prototypes = np.random.permutation(X)[:self.k].copy()
        return self
 
    def find_nearest_prototype(self, x):        
        dist = square_euclid(x, self.prototypes)       
        return np.argmin(dist)

    def fit(self, X):

        self.init_prototypes(X)
        self.errors = [self.score(X)]
        T = self.n_epochs * len(X)
        t = 0

        for epoch in range(self.n_epochs):
            for x in np.random.permutation(X):
                eta_t = self.eta * (self.etamin / self.eta) ** (t / T)
                lambda_t = self.lambda0 * (self.lambdamin / self.lambda0) ** (t / T)
                dists = square_euclid(x, self.prototypes)
                ranks = np.argsort(dists)

                for i, neuron_idx in enumerate(ranks):
                    h = np.exp(-i / lambda_t)
                    self.prototypes[neuron_idx] += eta_t * h * (x - self.prototypes[neuron_idx])
                t += 1
            self.errors.append(self.score(X))        
        return self


    def predict(self, X):
        
        return np.array([self.find_nearest_prototype(x) for x in X], dtype=np.int32)

    def score(self, X):
        
        error = []
        for x in X:
            dist = square_euclid(x, self.prototypes)
            m = np.argmin(dist)
            error.append(dist[m])
        return np.mean(error)
    
img = mpimg.imread('ssn-lab-06-neural-gas-BeProudPLS-master/dane/Lenna.png')

# Jeśli jest w formacie PNG, to może być typu float (0.0–1.0), a nie uint8
if img.dtype != 'uint8':
    print("Zmieniam")
    img = (img * 255).astype('uint8')

# przy n_prototypes=512 lambda0=5 lambdamin =0.1 n_epochs=10
#Średni błąd kwantyzacji: 28.954
#[MSE z funkcji] Średni błąd kwantyzacji: 29.286

# Rozwiązanie

patch_size=(3, 3)
n_prototypes = 512

X = img_to_vectors(img, patch_size).astype(np.float64)
X = X /255.0
vq = Gn(n_epochs=5, n_prototypes=n_prototypes, eta=0.1)
vq.etamin = 0.05
vq.lambda0 = 5
vq.lambdamin = 0.01
vq.fit(X)
pred = vq.predict(X)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(img)
axs[0].set_title('Oryginalny obraz')
axs[0].axis('off')
restored_img = vectors_to_image(vq.prototypes[pred] * 255, img_shape=img.shape, patch_size=patch_size)
axs[1].imshow(restored_img.astype(np.uint8))
axs[1].set_title(f'Księga kodów: {n_prototypes}\nMSE = {mse(img, restored_img):.2f}')
axs[1].axis('off')
axs[2].plot(range(len(vq.errors)), vq.errors, marker='o')
axs[2].set_title("Błąd kwantyzacji")
axs[2].set_xlabel("Epoka")
axs[2].set_ylabel("Błąd")
plt.tight_layout()
plt.show()