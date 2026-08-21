import cv2
import numpy as np

from pathlib import Path

images_path = Path(__file__).resolve().parents[1] / "simulations" / "interferograms"

file_name = "block_2.5_mm(simulated).png"
complete_path = images_path / file_name

if complete_path.exists():
    img_array = np.fromfile(complete_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        print(f"Image loaded successfully from path: {complete_path}")
    else:
        print("Error: The file was found, but OpenCV could not decode it as an image.")
else:
    print(f"Error: File not found at path: {complete_path}")
    
# 2. Definir un tamaño máximo para tu pantalla (ej. 1000 píxeles de ancho)
ancho_maximo = 1000
escala = 1.0

# Si la imagen es más grande que 1000px, calculamos cuánto hay que encogerla
if img.shape[1] > ancho_maximo:
    escala = ancho_maximo / img.shape[1]

# 3. Crear una copia pequeña de la imagen solo para mostrarla en pantalla
img_pantalla = cv2.resize(img, (0,0), fx=escala, fy=escala)

print("Abriendo ventana... Dibuja el rectángulo y presiona ENTER.")

# 4. Usar la herramienta de selección en la imagen pequeña
# (Dibuja con el ratón y luego presiona la tecla ENTER)
roi_pequeno = cv2.selectROI("Selecciona y presiona ENTER", img_pantalla)
cv2.destroyAllWindows()

# Extraer coordenadas pequeñas
x_p, y_p, w_p, h_p = roi_pequeno

if w_p == 0 or h_p == 0:
    print("No seleccionaste nada. Vuelve a intentar.")
else:
    # 5. ¡La Magia! Convertir las coordenadas de vuelta al tamaño gigante original
    x = int(x_p / escala)
    y = int(y_p / escala)
    w = int(w_p / escala)
    h = int(h_p / escala)

    print("\n--- TUS COORDENADAS PARA LA IMAGEN ORIGINAL ---")
    print(f"y1:y2 -> {y}:{y+h}")
    print(f"x1:x2 -> {x}:{x+w}")
    print(f"Código exacto para copiar en tu otro programa:")
    print(f"roi = img[{y}:{y+h}, {x}:{x+w}]")