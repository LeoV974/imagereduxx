import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ipywidgets as widgets
from ipywidgets import interact
img = Image.open("parrot.jpg").convert("L")
img_matrix = np.array(img, dtype=float)
U, s, Vt = np.linalg.svd(img_matrix, full_matrices=False)

def reconstruct_image(k):
    S = np.diag(s[:k])
    img_reconstructed = np.dot(U[:, :k], np.dot(S, Vt[:k, :]))
    plt.figure(figsize=(8, 6))
    plt.imshow(img_reconstructed, cmap='gray')
    plt.title(f"Reconstructed Image using {k} singular values")
    plt.axis('off')
    plt.show()

# Create a slider widget. The maximum is set to the number of singular values
interact(reconstruct_image, k=widgets.IntSlider(min=1, max=len(s), step=1, value=50))
