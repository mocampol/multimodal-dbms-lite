# Detección facial y embeddings

El extractor de [`FaceEmbeddingExtractor`](../src/index/multimedia/face_embedding_extractor.py)
usa **YuNet** para detectar rostros y obtener cinco puntos faciales, y **SFace**
para alinear cada rostro y producir un embedding de **128 dimensiones**. Cada
resultado incluye el rectángulo detectado, la confianza, el recorte alineado de
112 × 112 y el embedding `float32` normalizado L2.

Esta función solo convierte una imagen en detecciones y vectores: no asocia
personas, no genera etiquetas y no almacena datos. Las imágenes y embeddings
siguen siendo datos biométricos sensibles; aplicar la
[política de datos y consentimiento](./face_data_policy.md) antes de usar fotos
de personas.

## Modelos locales

Los modelos no se versionan en Git. Descárgalos a
`benchmarks/corpus/face/models/`, una ruta ya excluida del repositorio. Se
seleccionaron los modelos del [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
en la revisión `48b10fa505aeb51fbd127fc60a5c660040e823b7`:

| Archivo | Modelo y licencia del OpenCV Zoo | SHA-256 |
|---|---|---|
| `face_detection_yunet_2026may.onnx` | [YuNet, MIT](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet) | `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0` |
| `face_recognition_sface_2021dec.onnx` | [SFace, Apache-2.0](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface) | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

En macOS/Linux, desde la raíz del proyecto:

```bash
mkdir -p benchmarks/corpus/face/models
curl --fail --location --output benchmarks/corpus/face/models/face_detection_yunet_2026may.onnx \
  https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2026may.onnx
curl --fail --location --output benchmarks/corpus/face/models/face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
shasum -a 256 benchmarks/corpus/face/models/*.onnx
```

Confirma que los hashes coincidan con la tabla. El código requiere
`opencv-python-headless` (ya declarado por el proyecto); no descarga modelos
automáticamente. YuNet y SFace están disponibles en OpenCV 5, que es la versión
de OpenCV declarada actualmente.

## Uso

```python
from index.multimedia import FaceEmbeddingExtractor

extractor = FaceEmbeddingExtractor(
    "benchmarks/corpus/face/models/face_detection_yunet_2026may.onnx",
    "benchmarks/corpus/face/models/face_recognition_sface_2021dec.onnx",
)
faces = extractor.detect_and_extract("ruta/local/a/imagen.jpg")

for face in faces:
    print(face.bbox, face.confidence)
    print(face.aligned_crop.shape)  # (112, 112, 3)
    print(face.embedding.shape)     # (128,)
```

También acepta imágenes BGR `uint8` de NumPy y bytes codificados. Devuelve una
lista vacía si no detecta rostros; las rutas inexistentes, imágenes inválidas,
modelos faltantes y salidas incompatibles producen errores explícitos.

## Pruebas

Las pruebas unitarias no requieren los modelos. La prueba de integración
verifica la dimensión real de SFace si ambos archivos están en la ruta local
indicada; si no, se omite y muestra el motivo.
