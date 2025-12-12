---
title: "Visión por Computador – Proyecto Final"
layout: default
nav_order: 1
---

# Proyecto Final — Detección y Seguimiento de Vehículos con YOLO y Flujo Óptico

**Curso:** Visión por Computador – 3009228  
**Semestre:** 2025-02  
**Facultad de Minas, Universidad Nacional de Colombia**  
**Departamento de Ciencias de la Computación y de la Decisión**

---

## Descripción del proyecto

Este proyecto implementa un sistema completo de visión por computador que integra técnicas avanzadas de detección y seguimiento de objetos para resolver un problema práctico: **el conteo automático de vehículos en videos de tráfico**.

El proyecto está dividido en tres etapas principales:

1. **Detección de Objetos**: Implementación y configuración de un modelo YOLO v8 para detectar vehículos en cada fotograma del video. El modelo está pre-entrenado en el dataset COCO y puede detectar automóviles, motocicletas, autobuses y camiones.

2. **Seguimiento de Objetos**: Aplicación de técnicas de flujo óptico (Lucas-Kanade) para seguir vehículos entre múltiples fotogramas, manteniendo la identidad de cada objeto a lo largo del tiempo mediante asociación de detecciones usando Intersection over Union (IoU).

3. **Aplicación Práctica**: Implementación de un sistema de conteo automático de vehículos que cruzan una línea virtual en una intersección, similar a los sensores de lazo inductivo usados en semáforos reales.

### Características del Sistema

- **Detección en tiempo real**: YOLO v8 procesa frames a alta velocidad
- **Seguimiento robusto**: Flujo óptico mantiene identidad entre frames
- **Conteo preciso**: Algoritmo geométrico detecta cruces de línea virtual
- **Visualización completa**: Bounding boxes, trayectorias, y contador en tiempo real
- **Evaluación cuantitativa**: Métricas detalladas de rendimiento

### Dataset y Video de Prueba

El sistema fue probado con un video de tráfico:
- **Resolución**: 1920 × 1080 píxeles
- **FPS**: 24
- **Duración**: ~42 segundos (1001 frames)
- **Frames procesados**: 300 (muestra para análisis)

---

## Acceso al informe completo

[Ver Informe Final]({{ site.baseurl }}/informe.html)

---

## Tecnologías y Herramientas

- **Python 3.10+**
- **OpenCV (cv2)**: Procesamiento de imágenes, video y algoritmos de visión por computador (flujo óptico)
- **Ultralytics YOLO v8**: Modelo de detección de objetos en tiempo real
- **NumPy**: Operaciones numéricas y arrays multidimensionales
- **Matplotlib**: Visualización de datos, gráficas y frames procesados
- **Jupyter Notebooks**: Análisis interactivo y desarrollo

## Resultados Principales

El sistema logra:
- **4,274 detecciones** en 300 frames procesados
- **114 objetos únicos** seguidos a lo largo del tiempo
- **45 vehículos** contados que cruzaron la línea virtual
- **4.33 FPS** de procesamiento (razonable para análisis)
- **230.77 ms** promedio por frame

### Métricas de Rendimiento

| Métrica | Valor |
|---------|-------|
| Frames procesados | 300 |
| Detecciones totales | 4,274 |
| Promedio detecciones/frame | 14.25 |
| Objetos seguidos | 114 |
| Vehículos contados | 45 |
| FPS procesamiento | 4.33 |
| Tiempo/frame | 230.77 ms |

## Visualizaciones Generadas

El sistema genera múltiples visualizaciones explicativas:

- **Ejemplo de detección YOLO**: Comparación frame original vs. con detecciones
- **Explicación de IoU**: Visualización del concepto de Intersection over Union
- **Línea virtual**: Ejemplos de detección de cruce de línea
- **Diagrama del pipeline**: Flujo completo del sistema
- **Análisis de resultados**: Gráficas de rendimiento y estadísticas
- **Frames procesados**: Muestra de frames con detecciones y trayectorias

Todas las visualizaciones se guardan automáticamente en la carpeta `results/` y se incluyen en el informe.

## Créditos

**Desarrollado para:** Trabajo 4 - Detección y Seguimiento de Objetos  
**Curso:** Visión por Computador – 3009228  
**Universidad Nacional de Colombia – Facultad de Minas (2025-02)**

---

## Cómo usar el sistema

1. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar el notebook**:
   - Abrir `notebooks/1_yolo_objetos.ipynb`
   - Ejecutar las celdas en orden
   - Los resultados se guardan automáticamente en `results/`

3. **Ver resultados**:
   - Video procesado: `results/video_procesado.mp4`
   - Estadísticas: `results/estadisticas.json`
   - Visualizaciones: `results/*.png`

---

**Para más detalles, consulta el [Informe Completo]({{ site.baseurl }}/informe.html)**
