
from pattern_generator import run_simulation
import matplotlib.pyplot as plt

# Paste your predicted values here
Du = 0.11562
Dv = 0.06008
F = 0.02208
k = 0.03182

V = run_simulation(Du, Dv, F, k, steps=10000, size=128)

plt.imshow(V / V.max(), cmap='gray')  # You can use 'inferno' too
plt.axis('off')
plt.title("Predicted Turing Pattern")
plt.show()
