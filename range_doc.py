"""
chermin = Path.cwd() # Chemin courant 
path = r"C:\Users\kane_\OneDrive\Bureau\Projet IDI.txt"
print(path)

# Ecrire dans un fichier
with open(path, "w") as f:
    f.write("Bonjour\nAu revoir")
  

print(a.name) # Nom duc fichier courant
print(a.parent) # Dossier courant 
print(a.stem)
print(a.suffix) # Extension du fichier
print(a.suffixes)
print(a.exists())
print(a.is_dir()) # est-ce un dossier ?
print(a.is_file()) # est-ce un dossier ?

"""

import os
import pathlib
from pathlib import Path
from glob import glob
import shutil

extensions = {".pdf":"Documents",
                ".txt":"Documents",
                ".pptx":"Documents",
                ".csv":"Documents",
                ".xls":"Documents",
                ".docx":"Documents",
                ".zip":"Documents",
                ".html":"Documents",
                ".exe":"Exectutable",
                ".ipynb":"code",
                ".mp3":"Musique",
                ".wav":"Musique",
                ".flac":"Musique",
                ".mp4":"Videos",
                ".avi":"Videos",
                ".gif":"Videos",
                ".jpg":"Image",
                ".autres":"Divers",
                ".png":"Image",
                ".bmp":"Image"}

def trie(chemin_dossier):
  """
  Cette fonction permet de trier les fichiers contenus dans un dossier selon les associations suivantes :
  mp3, wav, flac : Musique
  avi, mp4, gif : Videos
  bmp, png, jpg : Images
  txt, pptx, csv, xls, odp, pages : Documents
  autres : Divers
  """

  # On recupère tous les fichiers dans le dossier courant
  fichiers = [f for f in chemin_dossier.iterdir() if f.is_file()]

  for fichier in fichiers: # on parcours tous les fichiers
        # On recupère les dossier cible à partir du dictionnaire
        dossier_cible = extensions.get(fichier.suffix, "Divers")
        # on concatène le dossier de base avec le dossier cible
        dossier_cible_absolu = chemin_dossier / dossier_cible
        # On crée le dossier cible s'il n'existe pas
        dossier_cible_absolu.mkdir(exist_ok=True)
        # On concatène le dossier cible avec le nom du fichier
        fichier_cible= dossier_cible_absolu / fichier.name
        # On deplace le fichier
        fichier.rename(fichier_cible)

p = Path('C:/Users/kane_/Downloads/sources/data')
chemin_telechargement = Path("C:/Users/kane_/Downloads")

trie(chemin_telechargement)


################################


