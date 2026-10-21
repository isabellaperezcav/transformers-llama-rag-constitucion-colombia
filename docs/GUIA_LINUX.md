# Guía rápida — ejecutar todo en Linux (RTX 3050)

## 1. Requisitos del sistema (una vez)
```bash
nvidia-smi                      # debe mostrar tu RTX 3050. Si falla, instala el driver PROPIETARIO de NVIDIA
python3 --version               # 3.10, 3.11 o 3.12  (TensorFlow 2.17 NO funciona con 3.13)
sudo apt install -y python3-venv python3-pip zstd curl      # Debian/Ubuntu/Mint
```
- **Portátil con gráfica híbrida (Intel + NVIDIA):** no importa para cálculo; CUDA usa la NVIDIA aunque la pantalla use la Intel.
- **Driver:** en Ubuntu, `ubuntu-drivers devices` y `sudo ubuntu-drivers install`, luego reinicia. Con *Secure Boot* activo hay que registrar la clave del módulo (te lo pide al instalar) o desactivarlo.

## 2. Pon la carpeta en el disco de Linux
Trabajar desde la partición de Windows (NTFS/OneDrive) es lento y a veces queda en solo lectura (por el *inicio rápido* de Windows). Cópiala a tu home:
```bash
cp -r "/media/$USER/<TU_PARTICION_WINDOWS>/Users/isape/OneDrive/Escritorio/IA Generativa/Entregable Transformers" ~/Entregable_Transformers
cd ~/Entregable_Transformers        # debe contener Constitución_2024.pdf y los 2 notebooks
mkdir -p documentos_rag             # aquí van Ley_1755_2015.pdf y Ley_1581_2012.pdf (con 1755 y 1581 en el nombre)
```
Los notebooks buscan solos la carpeta (la del notebook, `~/Entregable Transformers`, etc.). Si la tienes en otro sitio: `export ENTREGABLE_DIR="/ruta/a/la/carpeta"` antes de abrir VS Code/Jupyter.

## 3. Entorno de Python (una vez)
```bash
python3 -m venv ~/venv_ia && source ~/venv_ia/bin/activate
pip install --upgrade pip
pip install -r requirements.txt        # TensorFlow+CUDA, Chroma, Ollama, Gradio…
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"    # debe listar la GPU
```
Si imprime `[]`: revisa `nvidia-smi` y que el venv sea Python 3.10–3.12.

## 4. Ollama (Punto 3)
```bash
curl -fsSL https://ollama.com/install.sh | sh
systemctl status ollama                       # debe estar "active (running)"
```
Los modelos (`llama3.2:3b`, `bge-m3`) los descarga el propio notebook.

## 5. Ejecutar
```bash
source ~/venv_ia/bin/activate
cd ~/Entregable_Transformers && code .        # VS Code → abre el notebook → Select Kernel → venv_ia
```
(También sirve `pip install jupyterlab && jupyter lab`.)

- **Puntos 1 y 2:** `PERFIL = "auto"` detecta la GPU → perfil `gpu`, **3 semillas**. La primera celda debe decir `GPU: NVIDIA GeForce RTX 3050 … | perfil: gpu | semillas: 3`. Haz primero una corrida corta con `PERFIL = "prueba"` (unos minutos).
- **Punto 3:** pon las 2 leyes en `documentos_rag/` y ejecuta de arriba abajo.

## 6. Ojo con la memoria de la GPU (3050 = 4–6 GB)
Ollama mantiene el modelo cargado ~5 min tras usarlo y TensorFlow reserva memoria: **no ejecutes ambos a la vez**. Antes de entrenar:
```bash
ollama stop llama3.2:3b ; ollama stop bge-m3
```
- *Out of memory* en el entrenamiento → en `PERFILES["gpu"]` baja `batch` (char 128→64, word 64→32).
- ¿Quieres terminar antes? `N_SEEDS = 2` (o 1) en la celda de perfiles.
