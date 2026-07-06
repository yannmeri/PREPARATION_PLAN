# Utiliser une image Python officielle légère
FROM python:3.12.7

# Installer les dépendances système pour OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail dans le conteneur
WORKDIR /app

# Installer les packages Python requis
RUN pip install --no-cache-dir numpy opencv-python

# Copier notre script dans le conteneur
COPY convolutional_neural_network_exo.py .

# Créer le dossier où seront sauvegardés les résultats
RUN mkdir /app/output

# Commande exécutée au démarrage du conteneur
CMD ["python", "convolutional_neural_network_exo.py"]