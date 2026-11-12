# Tarea 2 Deep Learning: detección de objetos en FjordFish

## Instalación

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
venv\Scripts\python.exe -m pip install -r requirements.txt
```

El dataset original va en `dataset/FjordFish/`, junto a esta carpeta.

## Ejecución

Desde esta carpeta, en este orden:

```powershell
python eda.py                                                                   # particiones, copia reducida y EDA
python cnn_propia.py entrenar                                                   # CNN propia, desde cero
python yolo.py entrenar --pesos cero                                            # YOLO11s, desde cero
python yolo.py entrenar --pesos coco --epocas 60 --paciencia 15                 # YOLO11s, COCO, fine-tuning completo
python yolo.py entrenar --pesos coco --epocas 60 --paciencia 15 --congelar 11   # YOLO11s, COCO, backbone congelado
python faster_rcnn.py entrenar                                                  # Faster R-CNN, COCO
python fps.py                                                                   # velocidad de inferencia
python comparar.py                                                              # tabla y gráficos finales
```

Cada entrenamiento termina evaluando el mejor modelo en test. Los resultados quedan en `resultados/`.

## Diseño de la CNN propia

El backbone sigue la idea de ResNet-18 y está escrito a mano, con 4 etapas de 2 bloques residuales. Los atajos residuales permiten entrenar desde cero sin que se desvanezca el gradiente, y se eligió la variante más chica de ResNet porque el dataset tiene menos de 2000 imágenes de entrenamiento.

Sobre el backbone va una FPN que combina la información semántica de las capas profundas con el detalle espacial de las primeras, y entrega un mapa de 160×160 (stride 4). Esa resolución alta conserva los peces pequeños, que son muchos en el dataset.

El cabezal es estilo CenterNet: predice un mapa de calor con el centro de cada objeto por clase, y en cada centro el tamaño de la caja y un pequeño desfase. Al no usar anchors ni NMS tiene menos hiperparámetros y es más simple de entrenar desde cero. Como casi todas las celdas del mapa son fondo, se usa una pérdida focal para que no dominen el entrenamiento.

Los aumentos son escala, traslación, volteo horizontal y variación de color, porque los peces aparecen a distintas distancias y orientaciones y el color cambia con la turbidez del agua. No se usa volteo vertical, ya que bajo el agua arriba y abajo no son equivalentes. Para controlar el sobreajuste, que aparece pasada la época 100, se usan BatchNorm, weight decay y early stopping.
