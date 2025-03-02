from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
img = Image.open("parrot.jpg").convert("L")
img_matrix = np.array(img)
print("Image shape:", img_matrix.shape)
U, s, Vt = np.linalg.svd(img_matrix, full_matrices=False)
print(s.shape)
k = 50 

# Reconstruct the image using only the top k singular values
S = np.diag(s[:k])
img_reconstructed = np.dot(U[:, :k], np.dot(S, Vt[:k, :]))

# Plot the original and reconstructed images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_matrix, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Reconstructed Image (k={k})")
plt.imshow(img_reconstructed, cmap='gray')
plt.axis('off')
plt.show()
