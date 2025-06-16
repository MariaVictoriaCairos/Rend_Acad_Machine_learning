# Rend_Acad_Machine_learning
Predicción del rendimiento académico con Machine Learning

# Objetivo:
El conjunto de datos dataset_estudiantes.csv trata sobre el rendimiento académico de estudiantes.
 Variables objetivo:
 1.	Para regresión: nota_final (variable continua entre 0 y 100)
 2.	Para clasificación: aprobado (variable binaria: 1 si la nota es ≥ 60, 0 en caso contrario)
 

# Estructura de datos y tipo de variable: 

| Columna                     | Descripción                                                                         | Tipo       |
| --------------------------- | ----------------------------------------------------------------------------------- | ---------- |
| `nota_anterior`             | Nota que obtuvo el alumno en la convocatoria anterior.                              | Numérica   |
| `tasa_asistencia`           | Tasa de asistencia a clase en porcentaje.                                           | Numérica   |
| `horas_sueno`               | Promedio de horas que duerme el alumno al día.                                      | Numérica   |
| `edad`                      | Edad del alumno.                                                                    | Numérica   |
| `nota_final`                | Nota final del alumno en la convocatoria.                                           | Numérica   |
| `aprobado`                  | Indicador binario (`1`/`0`): `1` si el alumno está aprobado, `0` en caso contrario. | Numérica   |
| `nivel_dificultad`          | Dificultad percibida por el alumno para el estudio.                                 | Categórica |
| `tiene_tutor`               | Indica si el alumno tiene tutor (`Sí`/`No`).                                        | Categórica |
| `horario_estudio_preferido` | Horario de estudio preferido por el alumno.                                         | Categórica |
| `estilo_aprendizaje`        | Forma de estudio que emplea el alumno.                                              | Categórica |



El número de filas que tenemos es 1000, y el número de columnas es 10
# Estructura del proyecto
```
.
├── Data
│   ├── dataset_estudiantes_EDA.csv    # Datos resultantes del EDA
│   ├── dataset_estudiantes.csv        # Datos originales 
│   ├── df_clasificacion.csv           # Datos preparados para el modelo de regresión logística
│   └── df_regresion.csv               # Datos preparados para el modelo de regresión Lineal
├── Images                             # Carpeta con imagenes para el Readme
├── Modelo                             # Carpeta con los modelos entrenados
│   ├── modelo_clasificacion.pkl
│   └── modelo_regresion.pkl
├── Notebook                           # Carpeta con spcrips
│   ├── 01-EDA.ipynb
│   ├── 02-Preproceso.ipynb
│   ├── 03-Regresion.ipynb
│   └── 04-Clasificacion.ipynb
├── enviroment.txt                     # Archivo con las depecdencias instaladas en el entorno
├── README.md                          # Informe del proyecto
└── requerimientos.docx                # Requerimientos del proyecto

5 directories, 26 files

```

# Análisis exploratorio de datos

## **Detección de duplicados**
Dado que el conjunto de datos no incluye un identificador único para cada alumno, la estrategia empleada fue comprobar duplicados exactos sobre todas las columnas del DataFrame. El análisis arrojó 0 registros duplicados exactos, lo que confirma que no existen filas repetidas en el dataset.

## **Identificación de Nulos**
el resultado fue: 
| Columna                     | Valores faltantes |
| --------------------------- | ----------------- |
| horas\_sueno                | 150               |
| horario\_estudio\_preferido | 100               |
| estilo\_aprendizaje         | 50                |
| nota\_anterior              | 0                 |
| tasa\_asistencia            | 0                 |
| edad                        | 0                 |
| nivel\_dificultad           | 0                 |
| tiene\_tutor                | 0                 |
| nota\_final                 | 0                 |
| aprobado                    | 0                 |

## **Estadistica descriptiva de variables numéricas**

| Variable         |  count |  mean |   std |  min |   25% |   50% |   75% |   max | median |  moda |
| ---------------- | -----: | ----: | ----: | ---: | ----: | ----: | ----: | ----: | -----: | ----: |
| nota\_anterior   | 1000.0 | 69.89 | 14.69 | 30.0 | 59.88 |  70.0 | 80.12 | 100.0 |   70.0 | 100.0 |
| tasa\_asistencia | 1000.0 | 73.99 | 18.20 | 20.0 | 61.51 |  75.0 | 88.49 | 100.0 |   75.0 | 100.0 |
| horas\_sueno     |  850.0 |  7.01 |  1.44 |  4.0 |  6.00 |  7.02 |  8.02 |  10.0 |   7.02 |   4.0 |
| edad             | 1000.0 | 23.53 |  3.48 | 18.0 | 21.00 |  24.0 | 27.00 |  29.0 |   24.0 |  18.0 |
| nota\_final      | 1000.0 | 71.44 |  9.56 | 30.0 | 64.78 | 71.40 | 77.90 | 100.0 |  71.40 |  72.8 |
| aprobado         | 1000.0 |  0.90 |  0.30 |  0.0 |  1.00 |  1.00 |  1.00 |   1.0 |   1.00 |   1.0 |

- nota_anterior
    * Media y mediana muy cercanas (≈ 70), lo que indica una distribución bastante simétrica.
    * Desviación estándar moderada (≈ 14.7), con valores que van de 30 a 100.
    * Moda en 100 sugiere un subgrupo de estudiantes con calificación perfecta en la convocatoria previa.
- tasa_asistencia
    * Alta asistencia global: media ≈ 74 % y mediana en 75 %.
    * Desviación estándar de 18, con casos desde 20 % hasta el 100 %.
    * Moda en 100 %, reflejando muchos estudiantes con asistencia perfecta.
- horas_sueno
    * Promedio de sueño diario ≈ 7 horas, con dispersión reducida (std ≈ 1.4).
    * Valores mínimos de 4 h y máximos de 10 h.
    * Moda en 4 h, lo que indica cierto grupo de alumnos con muy pocas horas de descanso.
- edad
    * Edad promedio de 23.5 años, mediana de 24 y modo en 18, señalando una gran proporción de alumnos de primer curso.
    * Rango de 18 a 29 años, con std ≈ 3.5.
- nota_final
    * Media ≈ 71.4 y mediana muy cercana (71.4), distribución moderadamente simétrica.
    * Desviación estándar de 9.6, valores entre 30 y 100.
    * Moda en 72.8 apunta al valor más frecuente en la calificación final.
- aprobado
    * 90 % de los registros son “1” (aprobado), reflejando alta tasa de éxito académica.

En conjunto, los estudiantes muestran buenos resultados y altos niveles de asistencia, con variabilidad moderada en notas y horas de sueño, y una población muy concentrada en edades jóvenes (18–24 años).

**Relación de las estadísticas con los histogramas**

A continuación se relaciona cada histograma con los estadísticos previos, para comprobar que la forma de la distribución concuerda con la media, la mediana y el modo obtenidos:

![Histogramas variables numéricas](./images/HistNumerico.png)


| Variable             | Forma del histograma                                   | Relación con media/mediana/moda                                                                                                                     |
| -------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **nota\_anterior**   | Campana prácticamente simétrica alrededor de 70        | • Media = 69.9 y mediana = 70 coinciden en el centro. <br> • Modo = 100 aparece como cola derecha secundaria, reflejado en la barra final más alta. |
| **tasa\_asistencia** | Distribución sesgada a la derecha, con acúmulo en 100  | • Media = 74 y mediana = 75 quedan a la izquierda del pico de 100. <br> • Modo = 100 corresponde al gran “techo” que vemos en el histograma.        |
| **horas\_sueno**     | Ligera campana centrada en 7 h, con cola en 4 h y 10 h | • Media ≈ 7 h y mediana ≈ 7.02 h concuerdan con el pico central. <br> • Modo = 4 h se ve en la barra prominente a la izquierda de la gráfica.       |
| **edad**             | Barras casi uniformes entre 18 y 29, con pico en 18    | • Media = 23.5 y mediana = 24 reflejan la “meseta” central. <br> • Modo = 18 corresponde al primer y más alto de los contenedores.                  |
| **nota\_final**      | Distribución similar a la de nota\_anterior, simétrica | • Media = 71.4 y mediana = 71.4 coinciden en el centro de la campana. <br> • Modo = 72.8 se ve en la barra más frecuente alrededor de ese valor.    |
| **aprobado**         | Distribución fuertemente sesgada (casi todo 1)         | • Media = 0.9 y mediana = 1 muestran que la mayoría de barras están en 1. <br> • Modo = 1 coincide con el alto pico en esa categoría.               |


En todos los casos la forma de los histogramas confirma:

- Las medias y medianas caen en el “centro” más elevado de cada distribución.
- Las modas (valores más frecuentes) se corresponden con las barras más altas.

## **Estadística descriptiva de variables categóricas**

| Variable                        | count | unique | top    | freq |
| ------------------------------- | ----: | -----: | ------ | ---: |
| **nivel\_dificultad**           |  1000 |      3 | Medio  |  504 |
| **tiene\_tutor**                |  1000 |      2 | No     |  597 |
| **horario\_estudio\_preferido** |   900 |      3 | Noche  |  344 |
| **estilo\_aprendizaje**         |   950 |      4 | Visual |  363 |

Frecuencias de valores únicos por categoría
- Frecuencia de ```Nivel de Dificultad```
    | Nivel de Dificultad | Recuento |
    | ------------------- | -------: |
    | Medio               |      504 |
    | Fácil               |      313 |
    | Difícil             |      183 |

- Frecuencia de ```Tiene Tutor```
| Tiene Tutor | Recuento |
| ----------- | -------: |
| No          |      597 |
| Sí          |      403 |

- Frecuencia de ```Horario de Estudio Preferido```
| Horario de Estudio Preferido | Recuento |
| ---------------------------- | -------: |
| Noche                        |      344 |
| Tarde                        |      337 |
| Mañana                       |      219 |

- Frecuencia de ```Estilo de Aprendizaje```
| Estilo de Aprendizaje | Recuento |
| --------------------- | -------: |
| Visual                |      363 |
| Auditivo              |      254 |
| Kinestésico           |      178 |
| Lectura/Escritura     |      155 |

**Interpretación de la estadística descriptiva categórica**

![Gráficos de barra de variables categóricas](./images/HistCatego.png)

A continuación se presenta un breve análisis de las variables cualitativas del conjunto de datos, con el objetivo de entender las preferencias y características de los estudiantes en aspectos clave como la percepción de dificultad, el acompañamiento académico, sus hábitos de estudio y su estilo de aprendizaje. Estas métricas ofrecen una visión clara de la distribución de estas categorías, permitiendo identificar tendencias predominantes: 
- Nivel de dificultad: La mayoría (≈50 %) percibe el estudio como de dificultad “Medio”, seguido por “Fácil” (31 %) y “Difícil” (18 %).
- Tiene tutor: Un 60 % de los alumnos no cuenta con tutor, frente a un 40 % que sí lo tiene.
- Horario de estudio preferido: Predomina el turno de “Noche” (≈38 %), luego “Tarde” (≈36 %) y finalmente “Mañana” (≈24 %).
- Estilo de aprendizaje: El estilo “Visual” es el más común (≈38 %), seguido de “Auditivo” (≈27 %), “Kinestésico” (≈18 %) y “Lectura/Escritura” (≈16 %).

## **Análisis de correlación numérica**

![Matriz de correlación de datos numéricos](./images/Correlacion1.png)

Aunque la variable aprobado presenta una correlación muy elevada con nota_final, se trata en realidad de una etiqueta derivada (por ejemplo, aprobado = 1 si nota_final ≥ umbral). Por tanto, su vínculo es puramente tautológico y no aporta información adicional útil para la predicción.

**Predictores verdaderamente útiles**
1. nota_anterior (r ≈ 0.47)
2. tasa_asistencia (r ≈ 0.32)

En segundo orden, se podría considerar también:
- horas_sueno (r ≈ 0.08), si bien su señal es débil, puede interactuar con asistencia o dificultad.
- edad (casi nula r ≈ –0.01), normalmente la descartaríamos salvo que haya un efecto de madurez o seniority.

**Relaciones Cruzadas**
Variable objetivo numérica nota_final, generamos la dispersión:

![graficas de dispersión ](./images/disper.png)

### 1. nota_anterior vs nota_final
- **Tendencia positiva clara**: los puntos ascienden de izquierda a derecha, confirmando que mejores notas previas tienden a asociarse con mejores notas finales.  
- **Dispersión intermedia**: en el rango 60–80 de nota_anterior hay más variabilidad en nota_final (algunos suben mucho, otros bajan), lo que sugiere influencia de factores adicionales.  
- **Outliers alentadores y preocupantes**:  
  - Pocos alumnos con nota_anterior baja (<50) alcanzan nota_final >85 (mejora notable).  
  - Algunos con nota_anterior >85 caen por debajo de 60 (problemas puntuales).  

### 2. tasa_asistencia vs nota_final
- **Ligera correlación positiva**: a mayor asistencia, tiende a mejorar la nota_final, aunque con más ruido que en nota_anterior.  
- **“Techo” en alta asistencia**: muchos valores en asistencia ≈100 % se dispersan entre 65 y 90, sin garantía de “nota perfecta”.  
- **Baja asistencia (<50 %)**: asociado casi siempre a notas finales <70, señal de riesgo de fracaso.  

### 3. horas_sueno vs nota_final
- **Distribución casi plana**: apenas se aprecia pendiente; el número de horas de sueño (4–10 h) no parece influir linealmente en la nota_final.  
- **Concentración central**: la mayoría de puntos (5–9 h) se agrupan entre nota_final 60–85.  
- **Pocos extremos**: estudiantes con muy poco sueño (<5 h) o mucho (>9 h) salen tanto bien como mal, lo que refuerza su débil poder predictivo.  

### 4. edad vs nota_final
- **Sin tendencia marcada**: los valores de edad (18–29 años) muestran dispersión vertical casi constante; la edad por sí sola no explica la variación de la nota.  
- **Ligera concentración en jóvenes**: muchos 18–22 años abarcan toda la gama de la nota_final, igual que los mayores; no se observa “ventaja de madurez” clara.  

### 5. aprobado vs nota_final
- **Discreto pero tautológico**: los puntos para `aprobado = 1` (columna derecha) están todos en nota_final ≥ umbral, y `0` en la izquierda.  
- **Línea divisoria nítida**: evidencia el corte binario de la variable, sin aportar nada nuevo más allá de la regla de negocio que la define.  

---

## Conclusión rápida
- **Fuertes predictores**: `nota_anterior` (más claro) y, en menor grado, `tasa_asistencia`.  
- **Poco o nada predictivas**: `horas_sueno` y `edad`.  
- **Variable derivada**: `aprobado` solo refleja el umbral aplicado a `nota_final`.  


## **Análisis de correlación categórica**

### Interpretación de las Boxplots
![Boxplot categóricas ](./images/boxplot2.png.png)
A continuación un resumen de lo que muestran tus gráficos para cada variable categórica:

---

## 1. Nivel de dificultad

**Media de la nota:**
- **Fácil** tiene la media más alta (~72 pts),  
- **Medio** queda en segundo lugar (~71 pts),  
- **Difícil** es el más bajo (~69 pts).

**Distribución:**
- En “Fácil” y “Medio” se ve una distribución más apretada alrededor de la media (menos dispersa),  
- En “Difícil” la variabilidad es algo mayor y aparecen más alumnos con notas bajas y muy altas (colitas más largas en el violín).

> **Conclusión:** A medida que la dificultad sube, baja ligeramente el rendimiento medio y crece la dispersión de resultados.

---

## 2. Tener tutor

**Media de la nota:**
- Alumnos con tutor sacan de media ~75 pts,  
- Sin tutor ~69 pts.

**Distribución:**
- Quienes tienen tutor presentan una distribución más concentrada cerca de valores altos (menos outliers bajos),  
- Sin tutor muestran mayor cola hacia abajo (más estudiantes con notas bajas).

> **Conclusión:** Contar con tutor se asocia a una nota media superior y menos alumnos con calificaciones muy bajas.

---

## 3. Horario de estudio preferido

**Media de la nota:**
- **Tarde** (~72 pts) y **Mañana** (~71 pts) están muy parejos,  
- **Noche** ligeramente por debajo (~70 pts).

**Distribución:**
- En “Noche” hay algo más de variabilidad y más valores extremos bajos,  
- “Tarde” y “Mañana” muestran distribuciones más compactas.

> **Conclusión:** Estudiar por la tarde o mañana parece rendir un pelín mejor (y más estable) que de noche.

---

## 4. Estilo de aprendizaje

**Media de la nota:**
- **Lectura/Escritura** y **Kinestésico** rondan ~72 pts,  
- **Auditivo** ~71 pts,  
- **Visual** ~70 pts.

**Distribución:**
- Todos los estilos tienen forma de violín similar, con cuartiles cercanos y colas parecidas;  
- Quizá “Visual” muestra algo más de dispersión baja.

> **Conclusión:** El estilo de aprendizaje apenas influye en la nota media (diferencias de 1–2 pts) y todos presentan variabilidad similar.

---

## Resumen general

- **Tutor** y **nivel de dificultad** son los factores categóricos con mayor impacto: tutor sube la media ~6 pts y dificultad alta la baja ~3 pts.  
- **Horario** y **estilo** tienen efectos menores (<2 pts de diferencia) y distribuciones parecidas entre niveles.

> Estos hallazgos pueden guiarte para focalizar intervenciones (por ejemplo, ofrecer tutorías o ajustar la dificultad del material) antes que en cambiar el horario o estilo de estudio.


## Inconsistencias 

Aquí tienes un conjunto de reglas de validación para detectar posibles inconsistencias en el DataFrame:

- **Rangos de las notas**
  - `0 ≤ nota_anterior ≤ 100`
  - `0 ≤ nota_final ≤ 100`

- **Coherencia de la variable “aprobado”**
  - `aprobado == 1` sólo si `nota_final ≥ 60`
  - `aprobado == 0` sólo si `nota_final < 60`

- **Porcentaje de asistencia**
  - `0 ≤ tasa_asistencia ≤ 100`

- **Horas de sueño plausibles**
  - `0 ≤ horas_sueno ≤ 24`

## **Conclusión del control de calidad**  
Tras aplicar todas las reglas de validación —rangos de notas, coherencia del indicador de “aprobado”, porcentaje de asistencia y horas de sueño— no se han detectado discrepancias. El conjunto de datos cumple con los criterios de integridad y consistencia, por lo que está listo para continuar con el análisis exploratorio y el modelado predictivo.

Para construir un modelo que realmente aprenda de factores exógenos y generalice bien sobre datos nuevos, debes eliminar `aprobado` de tu conjunto de entrenamiento, porque:

- Es una etiqueta derivada de la propia `nota_final` (`aprobado = 1` si `nota ≥ 60`) y, por tanto, introduce fuga de información.  
- Inflaría artificialmente las métricas de tu modelo (como precisión o R²), pues le estarías dando la respuesta “casi lista” en lugar de las variables verdaderamente predictivas.  

# Preprocesamiento
## Gestión de nulos

- **Variables categóricas**  
  Se sustituyen los valores faltantes por la categoría genérica **`"Unknown"`**, asegurando que ninguna etiqueta válida se vea desplazada y permitiendo identificar explícitamente la ausencia de dato.

- **Variables numéricas**  
  Se imputan los nulos con la **mediana** de cada columna, una medida robusta frente a valores atípicos que preserva la distribución original de los datos.

> Esta estrategia de imputación mantiene la integridad del dataset y aplica un tratamiento adaptado al tipo de variable, sin necesidad de eliminar filas con información parcial.  

**Nulos en variables numéricas**
|                          |   # 0 |
|--------------------------|-------|
| horas_sueno              |   150 |
| horario_estudio_preferido|   100 |
| estilo_aprendizaje       |    50 |

**Nulos en variables categóricas**
|                          |   # 0 |
|--------------------------|-------|
| horario_estudio_preferido|   100 |
| estilo_aprendizaje       |    50 |

## Gestión outliers
**Columnas Numéricas**
![Outliers numéricos ](./images/OutliersNum.png)
Sólo en **tasa_asistencia** y **nota_final** hay casos aislados fuera de rango. Sin embargo, están dentro del 1 y el 100 cumpliendo con las reglas de consistencia de los datos. 

Aunque estos valores están dentro del rango válido (1–100) y no violan ninguna regla de consistencia, se considera lo siguiente:

1. **Relevancia y validez del dato**  
   - Si esos extremos (por ejemplo, asistencia <30 % o nota >95) reflejan situaciones reales—alumnos con asistencia prácticamente nula o rendimiento excepcional—entonces son informativos y **no deberían eliminarse**.  
   

2. **Impacto en el modelado**  
   - Los outliers pueden distorsionar modelos basados en medias o varianzas (p. ej. regresión lineal clásica).  
   - Si vas a usar algoritmos sensibles a extremos, considera **transformaciones** (log, raíz cuadrada) o **escalado robusto** (RobustScaler, winsorización).

3. **Opciones de tratamiento**  
   - **Winsorización**: limitar al percentil 1–99 para atenuar el efecto de extremos.  
   - **Imputación condicional**: reemplazar outliers por el valor del percentil 5/95, manteniendo la tendencia sin quitar la información.  
   - **Modelos robustos**: usar regresión con pérdida Huber o árboles, que manejan bien valores atípicos sin necesidad de filtros.

---

> **Recomendación general:**  
> - **No eliminar** estos casos de raíz si pueden ser reales.  
> - **Documentar** su existencia y, si es necesario, aplicar **técnicas de mitigación** (winsorizar, robust scaler) antes de entrenar modelos sensibles.  
> - En todos los casos, conservar la copia original del dato para posible análisis posterior.  

## Preparar los datos para la regresión.

Ahora preparamos el DataFrame para la regresión, asumiendo que la variable objetivo es **`nota_final`**. El flujo de preprocesamiento es el siguiente:

1. **Codificación de variables categóricas**  
   - Para la columna binaria **`tiene_tutor`**, aplicamos **One-Hot Encoding**.  
   - Para las variables con más de dos categorías —**`nivel_dificultad`**, **`horario_estudio_preferido`** y **`estilo_aprendizaje`**— usamos **Target Encoding**.

2. **Escalado de variables numéricas**  
   - Normalizamos todas las columnas numéricas con **MinMaxScaler**, dejando sus valores en el rango [0, 1].

3. **Guardado del DataFrame preprocesado**  
   - Finalmente, volcamos el resultado a disco (por ejemplo, con `df.to_csv("df_reg_preprocessed.csv", index=False)`) para su uso en el modelo.

Con este proceso obtenemos un conjunto de datos listo para entrenar nuestro modelo de regresión sin pérdida de información ni sesgos de escala.  


# Proceso de entrenamiento del modelo de regresión

1. **Carga de datos**  
   Importamos el DataFrame preprocesado que contiene las características y la variable objetivo `nota_final`.

2. **Definición de X e y**  
   - **y**: `nota_final`  
   - **X**: todas las columnas restantes, que ya incluyen variables numéricas escaladas y categóricas codificadas.

3. **División en entrenamiento y prueba**  
   Separamos el conjunto en un 80 % para entrenamiento y un 20 % para evaluación, obteniendo tamaños de (800, 11) y (200, 11) respectivamente.

4. **Entrenamiento del modelo**  
   Ajustamos un modelo de regresión lineal sobre los datos de entrenamiento.

5. **Validación inicial**  
   - Gráficos de dispersión entre valores reales y predichos.  
   - Comparación de distribuciones (histogramas/KDE).  
   - Análisis de residuos (residuos vs. predicciones y QQ-plot).  
   - Exploración de la importancia de cada característica (coeficientes).

6. **Cálculo e interpretación de métricas**  
   Calculamos R², MAE y RMSE tanto en entrenamiento como en prueba, para evaluar la calidad del ajuste y detectar posible sobreajuste.
    |  Conjunto | R<sup>2</sup> |  MAE | RMSE |
    | :-------: | :-----------: | :--: | :--: |
    | **Train** |      0.32     | 6.41 | 7.97 |
    |  **Test** |      0.33     | 6.04 | 7.41 |

**R<sup>2</sup> (~0.32–0.33)**  
El modelo explica aproximadamente el 32–33 % de la varianza en los datos tanto de entrenamiento como de prueba. Es un ajuste modesto: capta cierta relación entre las variables explicativas y la variable objetivo, pero deja gran parte de la variabilidad sin explicar.

**MAE (6.41 train vs. 6.04 test)**  
En promedio, la predicción se desvía de la realidad en unas 6 unidades en ambos conjuntos. La ligera mejora en test sugiere que el modelo no está sobreajustando los datos de entrenamiento.

**RMSE (7.97 train vs. 7.41 test)**  
La raíz del error cuadrático medio, que penaliza más los errores grandes, es también algo menor en test. La diferencia marginal entre train y test indica una buena generalización (poco sobreajuste).

---

  ### Conclusión  
  El modelo generaliza razonablemente bien (train vs. test casi iguales), pero su poder predictivo es limitado (R<sup>2</sup> bajo).


7. **Entrenamiento final**  
   Una vez validadas las métricas, reentrenamos el modelo sobre el conjunto completo para su uso en producción.

8. **Optimización**  
   Exploramos técnicas de regularización (Ridge, Lasso), ajuste de hiperparámetros (GridSearchCV/RandomizedSearchCV) y posibles transformaciones o ingeniería de características para mejorar la generalización.  
Los resultados de las métricas son :
| Modelo               | Train R² | Train MAE | Train RMSE | Test R² | Test MAE | Test RMSE |
|:--------------------:|:--------:|:---------:|:----------:|:-------:|:--------:|:---------:|
| **Linear (L0)**         |   0.32   |   6.41    |    7.97    |  0.33   |   6.04   |   7.41    |
| **Ridge (L2)**          |   0.32   |   6.41    |    7.97    |  0.33   |   6.04   |   7.41    |
| **Lasso (L1)**          |   0.32   |   6.42    |    8.00    |  0.34   |   5.98   |   7.36    |
| **ElasticNet (L1+L2)**  |   0.32   |   6.42    |    8.00    |  0.34   |   5.99   |   7.37    |


## Puntos clave de la interpretación

### Regularización vs. regresión pura
- **Ridge** ofrece exactamente los mismos resultados que la regresión lineal sin regularizar, lo que indica que la penalización L2 no aporta ganancia en este caso.  
- **Lasso** y **ElasticNet** (combinación de L1 y L2) consiguen una ligera mejora en test R² (pasan de 0.33 a 0.34) y reducen marginalmente el MAE y RMSE en test. Esto sugiere que algo de regularización L1 ayuda a desechar o atenuar variables poco útiles.

### Generalización
- La diferencia entre métricas de train y test es muy pequeña en todos los modelos, lo que indica que **no hay sobreajuste significativo**.

### Mejor modelo “final”
- Aunque la mejora es modesta, **Lasso** (o ElasticNet) muestra el mejor rendimiento en test (MAE≈5.98 y RMSE≈7.36). Además, puede producir un modelo más interpretable al forzar coeficientes a cero.  
- Si la interpretabilidad y la selección automática de variables importan, **Lasso** sería la elección recomendada; si no, la regresión lineal simple es prácticamente igual de eficaz.

### R² bajo–medio
- Un R² de ~0.33–0.34 indica que solo captura un tercio de la varianza. Para avanzar se podría:
  1. Incorporar nuevas features o interacciones.  
  2. Probar modelos no lineales (árboles, ensambles).  
  3. Refinar la calidad de los datos (feature engineering).

---

En resumen, los modelos regularizados (especialmente **Lasso**) aportan una pequeña ventaja en generalización y podrían simplificar la interpretación, pero el poder predictivo global sigue siendo limitado (R²≈0.34).


# Ciclos de entrenamiento realizados para optimizar el modelo de regresión lineal

1. **Entrenamiento inicial (sin tratamiento de outliers)**  
   En la primera iteración se ajustó el modelo lineal directamente, dado que los valores parecían estar dentro de rangos razonables. Aunque la correlación entre predicciones y valores reales fue buena (tendencia cercana a y = x), el modelo aún presentaba un margen de error moderado y mostraba un ligero sesgo en los extremos.  

   Para abordar esto, aplicamos el método Z-score para detectar outliers y los reemplazamos por la mediana de cada variable. El resultado fue una mejora clara en la distribución de errores y en la capacidad de predicción:  
   ![Comparativa valor real vs. predicción](./images/modelo1ValRealPredicciones.png)  
   ![Distribución posterior a la imputación](./images/modelo1Distribucion.png)  
   ![Análisis de residuos tras el tratamiento](./images/modelo1residuo.png)  

2. **Eliminación de la variable “aprobado”**  
   El gráfico de importancia de características mostró que `aprobado` era, con gran diferencia, la variable con mayor peso en el modelo. Dado que no aporta información adicional relevante y puede introducir sesgo, decidimos excluirla del dataset antes de volver a entrenar.  
   ![Importancia de características](./images/modelo1caracteristicas.png)  

3. **Prueba con regularización Lasso**  
   Finalmente, exploramos el uso de Lasso para reducir el número de variables y mejorar la generalización. Sin embargo, la métrica predictiva no mejoró de forma significativa, por lo que hemos optado por mantener el modelo lineal ajustado con los pasos anteriores.

# Clasificación con Regresión Logística

A continuación se presenta el procedimeinto para el entrenamiento de un modelo de clasificación para predecir la variable `aprobado`.

---

## 1. Carga del dataset preprocesado

- Importar el archivo `dataset_preprocesado.csv`.
- Separar las variables explicativas (`X`) de la variable objetivo (`y = aprobado`).

---

## 2. División en conjuntos de entrenamiento y prueba

- **Conjunto de entrenamiento**: 800 muestras (80 % del total).  
- **Conjunto de prueba**: 200 muestras (20 % del total).  
- Se aplica **estratificación** según la proporción de clases en `aprobado` para mantener el mismo balance en ambos conjuntos.

---

## 3. Entrenamiento del modelo

- Utilizar **Regresión Logística** con:
  - Regularización L2 (`penalty='l2'`).
  - Solver `liblinear`.
  - `class_weight='balanced'` para compensar posibles desbalances.
  - `random_state=42` para reproducibilidad.

---

## 4. Validación con matriz de confusión

- Obtener predicciones sobre el conjunto de prueba.
- Construir la **matriz de confusión** para visualizar:
  - Verdaderos positivos (VP)
  - Falsos positivos (FP)
  - Verdaderos negativos (VN)
  - Falsos negativos (FN)

La matriz de confusión resultante es:

   ![Importancia de características](./images/matrizconfusion.png)  

- **Verdaderos negativos (TN = 0):** de las 15 muestras realmente negativas, ninguna fue correctamente clasificada como 0.  
- **Falsos positivos (FP = 15):** las 15 muestras negativas fueron todas clasificadas incorrectamente como positivas.  
- **Falsos negativos (FN = 0):** no hay ejemplos reales positivos clasificados como negativos.  
- **Verdaderos positivos (TP = 185):** las 185 muestras positivas se detectaron correctamente.  

---

## 5. Cálculo de métricas de clasificación

| Conjunto | Accuracy | Precision | Recall | F1-score |
|:--------:|:--------:|:---------:|:------:|:--------:|
| **train** |   0.92   |   0.92    |  1.00  |   0.96   |
| **test**  |   0.96   |   0.96    |  1.00  |   0.98   |

**Interpretación de las métricas**

---

**Recall = 1.00 (100 %) en ambos conjuntos**  
- El modelo no pierde ningún caso positivo (aprobado = 1): no hay falsos negativos.  
- Indica que todas las muestras positivas se detectan correctamente.  

**Precision < 1 (92 % train / 96 % test)**  
- Hay algunos falsos positivos (casos negativos clasificados como positivos).  
- En train, el 8 % de las predicciones positivas fueron en realidad negativas; en test, este “error” baja al 4 %.  

**Accuracy alta (92 % train / 96 % test)**  
- Aproximadamente la misma tendencia: como la clase positiva es mayoritaria, clasificar “siempre” como positivo ya daría ~92 % de acierto.  
- El ligero aumento en test sugiere que el modelo generaliza bien… pero ojo: esta métrica oculta el mal desempeño sobre la clase negativa.  

**F1-score elevado (0.96 train / 0.98 test)**  
- Refleja el buen compromiso entre precision y recall para la clase positiva.  
- Sin embargo, no dice nada sobre la capacidad de detectar muestras negativas.  

---

### Conclusión y advertencia

El modelo tiene recall perfecto para la clase “aprobado = 1”, pero a costa de predecir muy pocos (o ningún) ejemplos como “no aprobado” (ver matriz de confusión).  
En escenarios desbalanceados es crucial complementar con métricas de la clase negativa (recall_negativo, specificity, balanced accuracy) o ajustar el umbral de decisión para recuperar sensibilidad sobre la clase minoritaria.  

---

## 6. Evaluación del desbalanceo

| Conjunto | Accuracy | Precision | Recall | F1-score |
|:--------:|:--------:|:---------:|:------:|:--------:|
| **train** |   0.92   |   0.93    |  0.92  |   0.90   |
| **test**  |   0.96   |   0.96    |  0.96  |   0.95   |

**Interpretación de las métricas tras balancear**

---

**Accuracy**  
- Permanece alta en ambos conjuntos (92 % en entrenamiento, 96 % en prueba), lo que indica un buen ajuste global.

**Precision (clase positiva)**  
- **Train: 0.93** → De cada 100 predicciones positivas, 93 son correctas.  
- **Test: 0.96** → Mejora ligera en generalización, menor proporción de falsos positivos.

**Recall (clase positiva)**  
- **Train: 0.92** → Ahora pierde un 8 % de los positivos reales.  
- **Test: 0.96** → Recupera el 96 % de los positivos reales en datos nuevos.  
- Esto confirma que el modelo ya no “predice siempre positivo” como antes (recall = 1), sino que discrimina mejor.

**F1-score**  
- **Train: 0.90**  
- **Test: 0.95**  
- Refleja el nuevo compromiso entre precision y recall: ligeros falsos positivos y falsos negativos, pero balanceados.

---

### Conclusión

Tras aplicar técnicas de balanceo, el modelo deja de tener recall perfecto (ahora detecta correctamente casi el 92 %–96 % de los aprobados), lo que es saludable: comienza a capturar también la clase negativa.  
La precision elevada y el F1-score robusto (≥ 0.90 en train y ≥ 0.95 en test) muestran que el modelo mantiene un sólido desempeño global, sin sacrificar excesivamente la detección de aprobados.  
En resumen, el balanceo ha corregido la “trampa” de predecir siempre la clase mayoritaria, logrando un modelo más justo y fiable para ambas clases.  

---

## 7. Importancia de las características

- Extraer coeficientes del modelo y tomar su valor absoluto.
- Ordenar variables según magnitud del coeficiente.
- Visualizar un diagrama de barras con la importancia de cada variable.

---

## 8. Entrenamiento final

- Reentrenar el modelo con **todos** los datos (entrenamiento + prueba) usando la misma configuración.
- Validar mediante **validación cruzada** (5 folds) y reportar:
  - Balanced Accuracy media.
  - Desviación estándar.

---

## Resultados y conclusiones

1. **Métricas en test**  
   - Accuracy: …  
   - Balanced Accuracy: …  
   - AUC-ROC: …  

2. **Variables más importantes**  
   - `variable_1`: coeficiente 0.45  
   - `variable_2`: coeficiente 0.32  
   - …  

3. **Desempeño final (CV)**  
   - Balanced Accuracy media: 0.XX ± 0.0X  

Con este pipeline completamos el entrenamiento, la evaluación y la interpretación de un modelo de regresión logística para la predicción de la variable `aprobado`.  

# Ciclos de entrenamiento realizados para optimizar el modelo de regresión Logística
1. En el primer ciclo de entrenamiento, el coeficiente de nota_final es abrumadoramente mayor que el de cualquier otra variable, lo que indica que el modelo se apoya casi exclusivamente en esa información para predecir la etiqueta aprobado. Sin embargo, nota_final no es una característica independiente: el objetivo aprobado se define precisamente como “nota_final ≥ 60”. Esto introduce un claro data leakage (fuga de información), ya que el modelo estaría aprendiendo directamente de la variable que determina la clasificación.

Por este motivo, confirmamos que es necesario eliminar la columna nota_final del DataFrame antes de volver a entrenar. De este modo, forzamos al modelo a basar sus predicciones en las demás características (asistencia, estilo de aprendizaje, horas de sueño, etc.), evaluando realmente su capacidad para generalizar y detectar patrones significativos sin recurrir a la variable calculada a posteriori.

Con esta limpieza de variables, reducimos la redundancia, evitamos el sobreajuste y obtenemos métricas (precision, recall, F1, etc.) que reflejan el rendimiento real del modelo sobre datos que no contienen fuga de la etiqueta.
![Caracteristicas regresión logística](./images/caracteristicasregresionlogist.png).

# Resultados tras eliminar la columna `nota_anterior`

## 1. Matriz de confusión

|                | Predicho 0 | Predicho 1 |
|:--------------:|:----------:|:----------:|
| **Verdadero 0**|      1     |     14     |
| **Verdadero 1**|      1     |    184     |

- **True Negatives (TN = 1)**: 1 ejemplo negativo correctamente clasificado.  
- **False Positives (FP = 14)**: 14 ejemplos negativos clasificados erróneamente como positivos.  
- **False Negatives (FN = 1)**: 1 ejemplo positivo clasificado erróneamente como negativo.  
- **True Positives (TP = 184)**: 184 ejemplos positivos correctamente clasificados.

**Sensibilidad (Recall clase 1)**  
\[
\frac{TP}{TP + FN} = \frac{184}{184 + 1} \approx 0{,}995
\]  
**Especificidad (Recall clase 0)**  
\[
\frac{TN}{TN + FP} = \frac{1}{1 + 14} \approx 0{,}067
\]

---

## 2. Métricas por conjunto

| Conjunto | Accuracy | Precision | Recall | F1-score |
|:--------:|:--------:|:---------:|:------:|:--------:|
| **train** |   0.90   |   0.90    |  1.00  |   0.94   |
| **test**  |   0.92   |   0.93    |  0.99  |   0.96   |

- **Recall = 1.00 (train) / 0.99 (test)**: el modelo casi no pierde positivos.  
- **Precision ≈ 0.90 / 0.93**: sigue habiendo falsos positivos, pero en test mejora ligeramente.  
- **Accuracy alta**: 90 % en train, 92 % en test, en línea con la prevalencia de la clase positiva.

---

## 3. Métricas ponderadas (weighted)

| Conjunto | Accuracy | Precision | Recall | F1-score |
|:--------:|:--------:|:---------:|:------:|:--------:|
| **train** |   0.90   |   0.89    |  0.90  |   0.85   |
| **test**  |   0.92   |   0.90    |  0.92  |   0.90   |

- Estas cifras combinan rendimiento en ambas clases:  
  - **Recall ponderado = 0.90 / 0.92** refleja que la clase negativa empieza a recuperarse (antes era casi 0).  
  - **F1-score ponderado = 0.85 / 0.90** muestra una ligera caída global respecto al modelo con `nota_anterior`, pero un comportamiento más equilibrado entre clases.

---

## 4. Importancia de características (coeficientes)

1. `tasa_asistencia`  
2. `nivel_dificultad`  
3. `horario_estudio_preferido`  
4. `horario_estudio_preferido_missing`  
5. `tiene_tutor_Sí`  
6. `horas_sueno`  
7. `edad`  
8. `estilo_aprendizaje_missing`  
9. `tiene_tutor_No`  
10. `estilo_aprendizaje`  

> **Nota:** sin la variable `nota_anterior`, el modelo recae en indicadores de asistencia y dificultad del curso como principales predictores.

---

## 5. Conclusiones

- Al eliminar `nota_anterior`, el rendimiento **global** (accuracy, precision, recall positivo) solo cae ligeramente, pero mejora la **capacidad de detectar la clase negativa** (especificidad pasa de 0 % a ≈ 6.7 % y weighted recall sube a 0.92).  
- El **F1-score ponderado** también se estabiliza alrededor de 0.90, evidenciando un modelo más justo entre ambas clases.  
- Las nuevas variables top (`tasa_asistencia`, `nivel_dificultad`, etc.) proporcionan información relevante y podrían aprovecharse para seguir mejorando la discriminación de la clase minoritaria.
