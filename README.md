# Transformers tipo LLaMA y RAG con Ollama sobre la Constitución de Colombia

Generación de texto con un **Transformer tipo LLaMA** (a nivel de carácter y de palabra) y un **RAG híbrido con Ollama** que responde con cita de artículo sobre la Constitución Política de Colombia, la Ley 1755 de 2015 y la Ley 1581 de 2012.

> Trabajo del curso **IA Generativa 2026‑2S** (UAO), basado en el material del Prof. Jesús A. López. Autores: Isabella Perez Caviedes y Luz Angela Carabali Mulato.

## Qué contiene

| Punto | Notebook | Qué hace |
|---|---|---|
| 1 y 2 | [`notebooks/01_02_llama_char_word.ipynb`](notebooks/01_02_llama_char_word.ipynb) | Entrena un Transformer **LLaMA** (RMSNorm + RoPE + SwiGLU + GQA) y un baseline **GPT** sobre la Constitución, a nivel de **carácter** y de **palabra**, con 3 semillas, y evalúa con BPC, perplejidad, top‑k, diversidad y memorización. |
| 3 | [`notebooks/03_rag_ollama.ipynb`](notebooks/03_rag_ollama.ipynb) | RAG con **Ollama** (`llama3.2:3b` + `bge-m3`), **Chroma + BM25**, recuperación híbrida balanceada por documento, abstención por umbral, citas por artículo e interfaz **Gradio**. |

## Resultados principales

Corrida en una RTX 3050 de 6 GB, TensorFlow 2.17, 3 semillas (media ± desviación en el conjunto de test).

**Generación de texto**

| Nivel | Métrica | LLaMA | GPT (baseline) | Lectura |
|---|---|---|---|---|
| Carácter | BPC | 1,254 ± 0,003 | 1,247 ± 0,006 | Empate técnico |
| Carácter | Parámetros | 1,60 M | 1,83 M | LLaMA usa menos |
| Carácter | val_loss tras la época 1 | 2,06 | 2,71 | LLaMA converge más rápido |
| Palabra | Perplejidad | 50,1 ± 1,4 | 57,1 ± 1,0 | LLaMA mejor, sensible a la regularización |
| Palabra | Accuracy top‑1 | 26,9 % | 24,6 % | |

El BPC de palabra (0,99) es **indicativo**: el 10 % de los tokens de test son `<unk>`, lo que abarata la predicción, así que no se compara con el de carácter.

**RAG** (23 preguntas con respuesta + 7 sin respuesta)

| Métrica | Valor |
|---|---|
| Recuperación: Hit@1 / Hit@5 / MRR | 0,74 / 0,91 / 0,81 |
| Dato correcto en la respuesta: con RAG vs sin RAG | **0,89 vs 0,32** |
| Fidelidad (juez LLM 1‑5): con RAG vs sin RAG | 3,3 vs 2,0 |
| Preguntas sin respuesta rechazadas | 5 de 7 |

Limitaciones conocidas: el modelo alucina en preguntas “cercanas” sin respuesta (p. ej. un “artículo 300 de la Ley 1581”), la recuperación falla con brecha léxica (“igualdad”, “presunción de inocencia”) y los artículos transitorios quedan etiquetados con el último artículo permanente.

## Estructura del repositorio

```
.
├── notebooks/            # Puntos 1‑2 y Punto 3
├── documentos_rag/       # PDF de las dos leyes (no versionados)
├── salidas/              # CSV y gráficas de resultados (los modelos .keras no se suben)
├── docs/                 # guía de instalación, diapositivas y guion
├── Constitución_2024.pdf # (no versionado) va en la raíz
├── requirements.txt
└── LICENSE
```

Los notebooks buscan solos la carpeta del proyecto (la del notebook o su carpeta padre). Si la tienes en otro sitio, define `ENTREGABLE_DIR`.

## Instalación (Linux con GPU NVIDIA)

Requisitos: driver propietario de NVIDIA (`nvidia-smi`), Python 3.10–3.12 y [Ollama](https://ollama.com).

```bash
git clone https://github.com/<tu-usuario>/transformers-llama-rag-constitucion-colombia.git
cd transformers-llama-rag-constitucion-colombia

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"   # debe listar la GPU

curl -fsSL https://ollama.com/install.sh | sh          # los modelos los descarga el notebook
```

Sin GPU, sustituye `tensorflow[and-cuda]` por `tensorflow` en `requirements.txt`: el notebook usa un perfil `cpu` ligero con 1 semilla. Más detalles en [`docs/GUIA_LINUX.md`](docs/GUIA_LINUX.md).

## Datos

1. Descarga la **Constitución Política de Colombia** (edición Corte Constitucional 2024) y guárdala como `Constitución_2024.pdf` en la raíz.
2. Descarga las **Leyes 1755 de 2015 y 1581 de 2012** y guárdalas en `documentos_rag/` con `1755` y `1581` en el nombre (ver [`documentos_rag/README.md`](documentos_rag/README.md)).

## Uso

1. Abre `notebooks/Punto1y2_LLaMA_Constitucion_LOCAL.ipynb`. Ejecuta primero con `PERFIL = "prueba"` (unos minutos) para verificar el entorno y luego con `"auto"` (detecta la GPU y usa 3 semillas; ≈ 1–1,5 h en una RTX 3050).
2. Abre `notebooks/Punto3_RAG_Ollama_Constitucion_LOCAL.ipynb` y ejecútalo de arriba abajo. Antes de entrenar, libera la GPU con `ollama stop llama3.2:3b && ollama stop bge-m3`: Ollama y TensorFlow no caben a la vez en 6 GB.
3. Los resultados quedan en `salidas/` (CSV y gráficas) y la interfaz Gradio en `http://127.0.0.1:7860`.

También se puede ejecutar en **Google Colab** (perfil `gpu`; el notebook del RAG instala Ollama y pide subir los PDF).

## Decisiones de diseño

- **Split por bloques intercalados** (90/5/5 en carácter, 80/10/10 en palabra) para que test sea representativo de toda la Constitución y no de sus artículos transitorios.
- **Vocabulario de palabras solo con entrenamiento** (sin fuga hacia validación/test) y texto en minúsculas con Python.
- **3 semillas** y un veredicto con umbral de ruido: una diferencia menor a dos desviaciones se declara empate.
- **Chunking por artículo**, recuperación **híbrida** (Chroma + BM25 con Reciprocal Rank Fusion) y **balanceada por documento**.
- **Evaluación del RAG** con dato clave, similitud, juez LLM, rúbrica manual y preguntas **sin respuesta** (lejanas y cercanas).

## Referencias

- Vaswani et al., *Attention Is All You Need* — https://arxiv.org/abs/1706.03762
- Touvron et al., *LLaMA* — https://arxiv.org/abs/2302.13971 · *Llama 2* — https://arxiv.org/abs/2307.09288
- Su et al., *RoFormer / RoPE* — https://arxiv.org/abs/2104.09864
- Shazeer, *GLU Variants Improve Transformer (SwiGLU)* — https://arxiv.org/abs/2002.05202
- Ainslie et al., *GQA* — https://arxiv.org/abs/2305.13245

## Licencia

Código bajo licencia [MIT](LICENSE). La Constitución y las leyes son textos públicos y no se redistribuyen en este repositorio.
