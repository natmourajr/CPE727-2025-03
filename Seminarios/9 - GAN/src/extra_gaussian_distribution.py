import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def gaussian(x, mean, var):
    return 1/np.sqrt(2*np.pi*var) * np.exp(-(x-mean)**2/(2*var))

x = np.linspace(-7, 7, 500)

configs = [((-1, 2), (3, 1)), ((-1, 2), (1, 1.5)), ((-1, 2), (-1, 2))]

file_paths = []

for idx, ((m1, v1), (m2, v2)) in enumerate(configs, 1):

    y1 = gaussian(x, m1, v1)
    y2 = gaussian(x, m2, v2)

    plt.figure(figsize=(8, 3))
    plt.plot(x, y1, label=f"mean={m1}, var={v1}")
    plt.plot(x, y2, label=f"mean={m2}, var={v2}")
    plt.legend()
    plt.grid(alpha=0.3)

    output_path = f"gaussian_plot_{idx}.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    file_paths.append(output_path)

imgs = [Image.open(p) for p in file_paths]

w = max(img.width for img in imgs)
h = sum(img.height for img in imgs)
final_img = Image.new("RGB", (w, h), (255, 255, 255))

current_y = 0
for img in imgs:
    final_img.paste(img, (0, current_y))
    current_y += img.height

final_img.save("gaussian_final.png")