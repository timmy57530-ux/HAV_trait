# -*- coding: utf-8 -*-
"""
fonctionTCi — version standalone (sans IPython / Analyse_Freq)
Compatible Anaconda / double-clic .bat
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import scipy.signal as sp
import csv
import os


##############___ Recupération, Reechantillonage, découpage, plots ___##############

def fs_detect(temps):
    freq=len(temps)/temps[-1]
    return (freq)



def detect_separator(file_path, skip_lines=0):
    """Détecte automatiquement le séparateur utilisé dans un fichier texte.
    
    Parameters
    ----------
    file_path : str
        Chemin du fichier à analyser.
    skip_lines : int, optional
        Nombre de lignes à ignorer avant détection (par défaut 0).
    
    Returns
    -------
    str or None
        Le séparateur détecté ou None si aucun n'est trouvé.
    """
    with open(file_path, "r", encoding="utf-8") as file:  # Modifier l'encodage si nécessaire
        # Ignorer les lignes d'en-tête
        for _ in range(skip_lines):
            next(file, None)

        # Lire la première ligne après le skip
        sample = file.readline()
        
        # Liste des séparateurs possibles
        for sep in ["\t", ";", ",", " "]:
            if sep in sample:
                return sep  # Retourne le premier séparateur trouvé

    return None  # Aucun séparateur détecté (fichier mal formaté ?)



def Recup(mesure, skip, colonnes_a_garder):
    """
    Charge un fichier de mesures et retourne les colonnes demandées.
    Détecte automatiquement le séparateur utilisé.

    Parameters
    ----------
    mesure : str
        Nom du fichier sans extension .txt.
    skip : int
        Nombre de lignes à ignorer en début de fichier.
    nb_colonnes : int
        Nombre total de colonnes dans le fichier.
    colonnes_a_garder : list
        Indices des colonnes à extraire.

    Returns
    -------
    list
        Liste des colonnes demandées sous forme de listes.
    """
    nb_colonnes=colonnes_a_garder[-1]+1
    data = [[] for _ in colonnes_a_garder]
    lien = f"{mesure}"
    # Détecter automatiquement le séparateur
    sep = detect_separator(lien, skip)
    if not sep:
        raise ValueError("Impossible de détecter le séparateur dans le fichier.")

    with open(lien, "r", encoding="utf8") as file:
        reader = csv.reader(file, delimiter=sep)  # Utilisation du bon séparateur

        # Ignorer les lignes d'en-tête
        for _ in range(skip):
            next(reader, None)

        for row in reader:
            if len(row) < nb_colonnes:
                continue  # Sauter les lignes incomplètes

            for idx, col in enumerate(colonnes_a_garder):
                try:
                    data[idx].append(float(row[col]))
                except ValueError:
                    print(f"Erreur de conversion : {row}")  # Debug
    return data


    
def Reech(signal, Fe, Fe_new):
    """
    Fonction pour rééchantillonner un signal à une nouvelle fréquence d'échantillonnage.

    Parameters
    ----------
    signal : list or np.array
        Le signal à rééchantillonner.
    Fe : int
        Fréquence d'échantillonnage du signal d'origine.
    Fe_nouveau : int
        Nouvelle fréquence d'échantillonnage souhaitée.

    Returns
    -------
    signal_resample : np.array
        Signal rééchantillonné à la nouvelle fréquence.
    """
    
    # Calcul du nombre de points dans le signal rééchantillonné
    N = len(signal)
    N_nouveau = int(N * Fe_new / Fe)  # Nouveau nombre de points en fonction des fréquences

    # Rééchantillonnage du signal à la nouvelle fréquence
    signal_resample = sp.resample(signal, N_nouveau)
    
    return signal_resample



def Cut(signal, Fe, début, fin, en_ms=True):
    """
    Fonction qui découpe un signal temporellement.

    Parameters
    ----------
    signal : list or np.ndarray
        Signal d'entrée à découper.
    Fe : int
        Fréquence d'échantillonnage de la mesure (Hz).
    début : float
        Temps de début du découpage (en secondes ou millisecondes).
    fin : float
        Temps de fin du découpage (en secondes ou millisecondes).
    en_ms : bool, optional
        True si début et fin sont en millisecondes, False s'ils sont en secondes. 
        Par défaut, True (valeurs en ms).

    Returns
    -------
    list or np.ndarray
        Signal découpé.
    """ 
    # Vérification si signal est vide
    if isinstance(signal, np.ndarray):
        if signal.size == 0:
            raise ValueError("Le signal est vide.")
    else:  # Si signal est une liste Python
        if not signal:
            raise ValueError("Le signal est vide.")

    if début >= fin:
        raise ValueError("Le temps de début doit être inférieur au temps de fin.")

    # Conversion en secondes si nécessaire
    if en_ms:
        début /= 1000
        fin /= 1000

    # Calcul des indices
    periode = 1 / Fe  # Période d'échantillonnage en secondes
    indice1 = max(0, min(len(signal), int(début / periode)))
    indice2 = max(0, min(len(signal), int(fin / periode)))

    return signal[indice1:indice2]




def Plot(signaux, Fe, temps=None, labels=None, couleurs=None, xlimup=None, xlimdwn=None, ylimup=None, ylimdwn=None):
    """
    Fonction qui trace plusieurs signaux sur un même graphique avec affichage des valeurs RMS dans la légende.

    Parameters
    ---------
    signaux : list of lists
        Liste des signaux à tracer (chaque signal est une liste de valeurs).
    Fe : int
        Fréquence d'échantillonnage du signal.
    temps : list, optional
        Liste des instants de temps (par défaut, sera générée automatiquement).
    labels : list, optional
        Liste des labels des signaux (pour la légende).
    couleurs : list, optional
        Liste des couleurs pour chaque signal.
    xlimup : float, optional
        Limite supérieure de l'axe X.
    xlimdwn : float, optional
        Limite inférieure de l'axe X.
    ylimup : float, optional
        Limite supérieure de l'axe Y (si non spécifié, calcul automatique).
    ylimdwn : float, optional
        Limite inférieure de l'axe Y (si non spécifié, calcul automatique).

    Returns
    -------
    None
    """    
    nb_signaux = len(signaux)
        # Génération automatique du temps si non spécifié
    if temps is None:
        periode = 1 / Fe
        temps = np.linspace(0, len(signaux[0]) * periode, len(signaux[0]))

    # Gestion des couleurs automatiques si non spécifiées
    if couleurs is None:
        couleurs = plt.cm.viridis(np.linspace(0, 1, nb_signaux))  # Dégradé de couleurs

    # Calcul et mise à jour des labels avec la valeur RMS
    if labels is None:
        labels = [f"Signal {i+1}" for i in range(nb_signaux)]
    
    labels = [f"{label} (RMS: {round(np.sqrt(np.mean(np.square(signal))),2)} m/s²)" for label, signal in zip(labels, signaux)]

    # Calcul des limites Y si non spécifiées
    if ylimup is None or ylimdwn is None:
        all_values = np.concatenate(signaux)
        ylimup = np.max(all_values) * 1.1 if ylimup is None else ylimup
        ylimdwn = np.min(all_values) * 1.1 if ylimdwn is None else ylimdwn

    # Tracé des signaux
    plt.figure(figsize=(10, 5))
    for i, signal in enumerate(signaux):
        plt.plot(temps, signal, label=labels[i], color=couleurs[i])

    # Paramètres du graphique
    plt.title("Tracé de plusieurs signaux")
    plt.xlabel("Temps (en s)")
    plt.ylabel("Amplitude (en m/s²)")
    plt.xlim(xlimdwn if xlimdwn is not None else temps[0], xlimup if xlimup is not None else temps[-1])
    plt.ylim(ylimdwn, ylimup)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def PlotFreq(signaux, Fe, labels=None, couleurs=None, xlimup=None, xlimdwn=0.1, ylimup=None, ylimdwn=0):
    """
    Trace plusieurs signaux en domaine fréquentiel avec :
    - Un axe X logarithmique (fréquence)
    - Un axe Y linéaire (amplitude)

    Parameters
    ----------
    signaux : list of lists
        Liste des signaux à analyser (chaque signal est une liste de valeurs).
    Fe : int
        Fréquence d'échantillonnage du signal.
    labels : list, optional
        Labels des signaux pour la légende.
    couleurs : list, optional
        Couleurs pour chaque signal.
    xlimup : float, optional
        Limite supérieure de l'axe X (fréquence max).
    xlimdwn : float, optional, default=0.1
        Limite inférieure de l'axe X (fréquence min), évite log(0).
    ylimup : float, optional
        Limite supérieure de l'axe Y (amplitude max). Si None, prend max du spectre +10%.
    ylimdwn : float, optional, default=0
        Limite inférieure de l'axe Y (amplitude min).

    Returns
    -------
    tuple
        (freqs, spectres)
    """    
    nb_signaux = len(signaux)

    # Gestion des couleurs et labels automatiques
    couleurs = couleurs or plt.cm.viridis(np.linspace(0, 1, nb_signaux))
    labels = labels or [f"Signal {i+1}" for i in range(nb_signaux)]

    spectres = []

    # Calcul FFT via numpy (remplacement de QP.TFD)
    def _fft_spectrum(sig, fe):
        n = len(sig)
        freqs = np.fft.rfftfreq(n, d=1.0/fe)
        spectrum = np.abs(np.fft.rfft(sig)) * 2.0 / n
        return freqs, spectrum

    freqs, _ = _fft_spectrum(np.array(signaux[0]), Fe)

    # Initialisation du graphique
    plt.figure(figsize=(10, 5))

    # Tracé des spectres
    max_amplitude = 0
    for i, signal in enumerate(signaux):
        freqs_i, spectrum = _fft_spectrum(np.array(signal), Fe)
        max_amplitude = max(max_amplitude, np.max(spectrum))
        plt.semilogx(freqs_i, spectrum, label=labels[i], color=couleurs[i])
        spectres.append(spectrum)

    # Définition de ylimup automatique si None
    if ylimup is None:
        ylimup = max_amplitude * 1.10  # Ajout de 10%

    # Paramètres du graphique
    plt.title("Spectre fréquentiel des signaux")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude (en m/s²)")

    # Définition des axes
    plt.xscale('log')  # Axe X en logarithmique
    plt.xlim(xlimdwn, xlimup if xlimup is not None else max(freqs))
    plt.ylim(ylimdwn, ylimup)

    # Formatage des ticks X en notation normale (10, 100, 1000...)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    # Affichage de la légende et des grilles
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return freqs, spectres



def PlotDSP(signaux, Fe, labels=None, couleurs=None, xlimup=None, xlimdwn=0.1, ylimup=None, ylimdwn=0):
    """
    Trace plusieurs signaux en domaine fréquentiel avec :
    - Un axe X logarithmique (fréquence)
    - Un axe Y linéaire (amplitude)
    """
    nb_signaux = len(signaux)
    couleurs = couleurs or plt.cm.viridis(np.linspace(0, 1, nb_signaux))
    if labels is None:
        labels = [f"Signal {i+1}" for i in range(nb_signaux)]

    # Initialisation
    spectres = []
    freqs = None
    energies = []

    # Calcul des spectres et de l’énergie associée
    for signal in signaux:
        f, Pxx = sp.welch(signal, Fe, nperseg=1024)
        if freqs is None:
            freqs = f
        spectres.append(Pxx)
        energies.append(np.trapz(Pxx, f))  # Aire sous la courbe = énergie

    # Mise à jour des labels avec énergie
    labels = [f"{label} (Énergie: {round(energie, 4)} m²/s⁴/Hz)" for label, energie in zip(labels, energies)]

    # Tracé
    plt.figure(figsize=(10, 5))
    max_amplitude = 0
    for spectrum, label, couleur in zip(spectres, labels, couleurs):
        plt.semilogx(freqs, spectrum, label=label, color=couleur)
        max_amplitude = max(max_amplitude, np.max(spectrum))

    # Limites des axes
    if ylimup is None:
        ylimup = max_amplitude * 1.10

    plt.xscale('log')
    plt.xlim(xlimdwn, xlimup if xlimup else max(freqs))
    plt.ylim(ylimdwn, ylimup)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    plt.title("DSP signaux")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("DSP (m²/s⁴/Hz)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

    return freqs, spectres



##############___ Filtres ___##############


def Filtre(signal, Fe, Coefs):
    """
    Fonction qui applique un flitre au signal donné de fréquence d'échantillonage Fe.
    Le filtre peut etre défini en fonction des coefficients de Laplace (Coefs) ou selon la liste de filtres prédéfinis.
    Si Coefs donnés, rappel : Bp/Ap, avec dans l'ordre [...], s², s, s**0'
    Parameters
    ----------
    signal : TYPE Liste
        Signal d'entrée à filtrer.
    Fe : TYPE float
        Fréquence d'échantillonage.
    Coefs : TYPE Liste ou string
        coefficients de Laplace (entrez [[Ap], [Bp]] ou type de filtre (wk, wd, wf, wc, we, wj, wsl, wsv, flat)
% - 'wk'  : seated - vertical
% - 'wd'  : seated - horizontal
% - 'wf'  : motion sickness  
% - 'wc'  : seat-back measurement
% - 'we'  : rotational vibration
% - 'wj'  : recumbent - under the head
% - 'wsl' : standing  - lateral 
% - 'wsv' : standing  - vertical 
% - 'flat'  : 
% - 'wh' :  
% - 'wb' : 
                                                        
                                 
    Returns
    -------
    signal filtré.

    """
    if Coefs == "wk":
        # Filtre de type 'wk'
        Bp = [1270465375.49280,120571885863.767,1914545634688.76,22126317686949.2,0,0]
        Ap = [40.9744578038939,42610.4727028893,22097442.1254741,2822859823.23484,170805432573.459,3890222906392.01,56957141058840.6,177979858721527,279242672043844]
    elif Coefs == "wd":
        if Fe <800 or Fe>4960 : 
            print("ERREUR : fréquence d'échantillonage hors scope (plage de Fs : [800Hz-4960Hz])")
        # Filtre de type 'wd'
        Bp = [39275345.5049098,493548547.621702,0,0]
        Ap = [7.91681348704628,7220.74756410636,3292615.99484951,75110048.5376594,739715022.917670,2155024641.74191,3117522507.36538]
    elif Coefs == "wf":
        if Fe <6 or Fe>57 : 
            print("ERREUR : fréquence d'échantillonage hors scope (plage de Fs : [6Hz-57Hz])")
        # Filtre de type 'wf'
        Bp = [21.2793821945601,10.4454923086499,3.28154818999833,0,0]
        Ap = [0.550400000000000,4.91000793049325,22.3880844812168,51.9022368613421,77.0584675564220,67.8854583205034,38.4034423914722,12.5240138255072,2.12255260399333]
    elif Coefs == "wc":
        if Fe <800 or Fe>9000 : 
            print("ERREUR : fréquence d'échantillonage hors scope (plage de Fs : [800Hz-9000Hz])")
        # Filtre de type 'wc'
        Bp = [628405528.078557,31587107047.7889,0,0]
        Ap = [31.6672539481851,30777.9543014346,14936030.3314812,1121457878.89607,35478769037.2373,119019896414.437,199521440471.385]
    elif Coefs == "we":
        if Fe <800 or Fe> 3775 : 
            print("ERREUR : fréquence d'échantillonage hors scope (plage de Fs : [800Hz-3775Hz])")        
        # Filtre de type 'we'
        Bp = [9818836.37622745,61693568.4527127,0,0]
        Ap = [3.95840674352314,3570.89536444882,1610619.26653097,21426392.4407048,127676137.108205,318601206.928794,389690313.420673]
    elif Coefs == "wj":
        if Fe <800 or Fe> 7200 : 
            print("ERREUR : fréquence d'échantillonage hors scope (plage de Fs : [800Hz-7200Hz])")        
        # Filtre de type 'wj'
        Bp = [326920.776181684,8464713.53372185,181495053.641829,0,0]
        Ap = [0.828100000000000,769.191750015778,357603.614719333,14096949.2529000,413125776.870074,1379362104.73202,2307306247.60736]
    elif Coefs == "wsl":
        # Filtre de type 'wsl'
        Bp = [1233871369.05425,34886891056.4607,97422578355.1682,0,0]
        Ap = [49.7428061814904,69487.1649387894,42250132.6212061,10110387651.7008,56812628627.5108,135253970655.532,81795118239.1377,24614971087.3508]
    elif Coefs == "wsv":
        # Filtre de type 'wsv'
        Bp = [1897086466.43271,235382233239.270,5164186255600.88,70682864098913.3,295669549292988,0]
        Ap = [47.7999605428994,52645.4417610091,28689114.9910251,4729896813.01352,327082968060.737,8628046037421.74,149068433096067,475520083514373,760932994211068]
    elif Coefs == "wp":
        if Fe <20001 or Fe> 100000 : 
            print("ERREUR : fréquence d'échantillonage hors scope (plage de Fs : ]20kHz-100kHz])")   
        # Filtre de type 'flat'
        Bp = [1,0,0]
        Ap = [1.5982336242639856e-07, 0.0005854095026181758, 1.1027493331569642, 186.18010380282874, 15716.641130400114]
    elif Coefs == "wh":
        # Filtre de type 'wh'
        Bp = [32124178431.518127, 3212318070714.359, 0, 0]
        Ap = [5.13456096, 58291.60705191402, 333487843.8061613, 69298071294.02808, 6566226006185.949, 259176019600812.75, 5049367681564479.0]
    elif Coefs == "flat":
        # Filtre de type 'flat'
        Bp = [1,0,0]
        Ap = [1.6211389382774044e-08, 0.00018070497595461168, 1.0101054016000002, 56.26246585890022, 1566.898394716946]  
    elif type(Coefs) == str :
        raise ValueError("Type de filtre non reconnu. Choisissez parmi 'wp', 'wc', 'wd','we', 'wf', 'wh', 'wj','wk' ou 'flat'.")

    else:
        # Si les coefficients Laplace sont donnés directement
        Bp, Ap = Coefs

    Bz, Az = sp.bilinear(Bp, Ap, Fe)
    
    signal_filtre =sp.lfilter(Bz,Az,signal)
    return(signal_filtre)


##############___ CALCULS VIB ___##############

def RMS(signal, Fe):
    """
    Calcule la valeur RMS (Root Mean Square) d'un signal, en prenant en compte le facteur de normalisation pour le calcul de A8.

    Parameters
    ----------
    signal : list or numpy array
        Le signal d'entrée pour lequel la valeur RMS doit être calculée.
    Fe : float
        La fréquence d'échantillonnage du signal en Hz.

    Returns
    -------
    float
        La valeur RMS du signal, normalisée en fonction de la durée du signal (en heures).
    """
    rms = round(np.sqrt(np.mean(np.square(signal))))
    return (rms)


def A_8_HAV(signal_x, signal_y, signal_z, Fe):
    """
    Calcule la valeur A8 pour les vibrations mains-bras (HAV) selon la norme ISO 5349.
    La valeur A8 est calculée en prenant la valeur maximale des valeurs RMS pondérées 
    pour les directions X, Y et Z après filtrage de pondération appropriée (Wh).

    Parameters
    ----------
    signal_x : list or numpy array
        Le signal de vibration mesuré sur l'axe X.
    signal_y : list or numpy array
        Le signal de vibration mesuré sur l'axe Y.
    signal_z : list or numpy array
        Le signal de vibration mesuré sur l'axe Z.
    Fe : float
        La fréquence d'échantillonnage du signal en Hz.

    Returns
    -------
    float
        La valeur A8 des vibrations mains-bras, prenant la valeur maximale parmi les directions X, Y et Z.
    """
    # Appliquer la pondération pour les vibrations mains-bras (selon la norme ISO 5349)
    signal_x = Filtre(signal_x, Fe, 'wh')  # Filtrage pondéré pour la direction X
    signal_y = Filtre(signal_y, Fe, 'wh')  # Filtrage pondéré pour la direction Y
    signal_z = Filtre(signal_z, Fe, 'wh')  # Filtrage pondéré pour la direction Z
    
    # Calculer les valeurs RMS pour chaque direction après filtrage
    RMS_x = RMS(signal_x, Fe)
    RMS_y = RMS(signal_y, Fe)
    RMS_z = RMS(signal_z, Fe)
    
    # Calculer la valeur A8 en prenant la valeur maximale parmi les directions
    A_8 = np.sqrt(RMS_x**2 + RMS_y**2 + RMS_z**2)
    return A_8


def A_8_WBV(signal_x, signal_y, signal_z, Fe):
    """
    Calcule la valeur A8 pour les vibrations du corps entier (WBV) selon la norme ISO 2631.
    La valeur A8 est calculée en utilisant une somme quadratique des valeurs RMS pondérées 
    pour les directions X, Y et Z après filtrage avec des pondérations spécifiques 
    (Wd pour X et Y, Wk pour Z).

    Parameters
    ----------
    signal_x : list or numpy array
        Le signal de vibration mesuré sur l'axe X.
    signal_y : list or numpy array
        Le signal de vibration mesuré sur l'axe Y.
    signal_z : list or numpy array
        Le signal de vibration mesuré sur l'axe Z.
    Fe : float
        La fréquence d'échantillonnage du signal en Hz.

    Returns
    -------
    float
        La valeur A8 des vibrations du corps entier, calculée comme la somme quadratique des RMS pondérés pour chaque direction.
    """
    # Appliquer la pondération pour les vibrations du corps entier (selon la norme ISO 2631)
    signal_x = Filtre(signal_x, Fe, 'wd')  # Filtrage pondéré pour la direction X
    signal_y = Filtre(signal_y, Fe, 'wd')  # Filtrage pondéré pour la direction Y
    signal_z = Filtre(signal_z, Fe, 'wk')  # Filtrage pondéré pour la direction Z
    
    # Calculer les valeurs RMS pour chaque direction après filtrage
    RMS_x = RMS(signal_x, Fe)
    RMS_y = RMS(signal_y, Fe)
    RMS_z = RMS(signal_z, Fe)
    
    # Calcul de A8 en prenant la somme quadratique des RMS pondérés
    A_8 = max ([(1.4 * RMS_x), (1.4 * RMS_y) , (RMS_z)])
    return A_8


def A_8_HTS(signal_x, signal_y, signal_z, Fe):
    """
    Calcule la valeur A8 pour les vibrations avec un filtrage plat (HTS), en prenant la valeur maximale 
    des valeurs RMS des trois directions X, Y et Z après application d'un filtrage sans pondération spécifique.

    Parameters
    ----------
    signal_x : list or numpy array
        Le signal de vibration mesuré sur l'axe X.
    signal_y : list or numpy array
        Le signal de vibration mesuré sur l'axe Y.
    signal_z : list or numpy array
        Le signal de vibration mesuré sur l'axe Z.
    Fe : float
        La fréquence d'échantillonnage du signal en Hz.

    Returns
    -------
    float
        La valeur A8 des vibrations HTS, prenant la valeur maximale parmi les directions X, Y et Z après filtrage plat.
    """
    # Appliquer un filtrage plat pour chaque direction
    signal_x = Filtre(signal_x, Fe, 'flat')
    signal_y = Filtre(signal_y, Fe, 'flat')
    signal_z = Filtre(signal_z, Fe, 'flat')
    
    # Calculer les valeurs RMS pour chaque direction après filtrage
    RMS_x = RMS(signal_x, Fe)
    RMS_y = RMS(signal_y, Fe)
    RMS_z = RMS(signal_z, Fe)
    
    # Calculer la valeur A8 en prenant la valeur maximale parmi les directions
    A_8 = np.sqrt(RMS_x**2 + RMS_y**2 + RMS_z**2)
    return A_8


def Sinus_Gen(Amp, Freq, Fs, Lien=None, Durée=60, Phase=0):   
    """
    Fonction qui génère un signal sinusoïdal et l'enregistre dans un fichier texte ou le retourne sous forme de liste.

    Parameters
    ----------
    Amp : float
        Amplitude du signal sinusoïdal.
    Freq : float
        Fréquence du signal en Hertz.
    Fs : float
        Fréquence d'échantillonnage en Hertz.
    Lien : str, optional
        Destination du fichier où enregistrer les données. Si None, retourne une liste.
    Durée : float, optional
        Durée du signal en secondes. The default is 60.
    Phase : float, optional
        Déphasage du signal en radians. The default is 0.

    Returns
    -------
    list or str
        Liste des valeurs (temps, signal) si aucun fichier n'est demandé, sinon chemin du fichier généré.
    """
    t = np.arange(0, Durée, 1/Fs)  # Génération des instants de temps
    signal = np.random.normal(0, 1, len(t)) # Amp * np.sin(2 * np.pi * Freq * t + Phase)  # Génération du signal sinusoidal
    
    if Lien is None:
        return (list(t), list(signal)) # Retourne la liste des paires (temps, valeur du signal)
    
    # Création du nom de fichier
    filename = f"sin(A{Amp}_F{Freq}_Fs{Fs}_D{Durée}_P{Phase}).txt"
    filepath = os.path.join(Lien, filename)
    
    # Écriture dans le fichier
    with open(filepath, "w") as f:
        for time, value in zip(t, signal):
            f.write(f"{time:.6f};{value:.6f}\n")
    
    return filepath







# Coefficients du filtre (exemples, à remplacer par les coefficients de Laplace du filtre souhaité)

#f1,f2,f3,f4,Q4,f5, Q5, f6, Q6
pflat=[0.4, 100, None, None, None, None, None, None, None]
pwc=[0.4, 100, 8, 8, 0.63, None, None, None, None]
pwd=[0.4, 100, 2, 2, 0.63, None, None, None, None]
pwe=[0.4, 100, 1, 1, 0.63,None, None, None, None]
pwf=[0.08, 0.63, None, 0.25, 0.86, 0.0625, 0.8, 0.1, 0.8]
pwj=[0.4, 100, None, None, None, 3.75, 0.91, 5.32, 0.91]
pwk=[0.4, 100, 12.5, 12.5, 0.63, 2.37, 0.91, 3.35, 0.91]



def Hipass(pond):
    f1=pond[0]
    w1=2*np.pi*f1
    a1=w1**2
    a2= np.sqrt(2)*w1
    a3=1
    return ([1,0,0],[a3,a2,a1])

def Lopass(pond):
    f2=pond[1]
    w2=2*np.pi*f2
    a1=1
    a2= np.sqrt(2)/w2
    a3=1/(w2**2)
    return ([1],[a3,a2,a1])

def TAC(pond):
    f3=pond[2]
    f4=pond[3]
    Q4=pond[4]
    w3=2*np.pi*f3
    w4=2*np.pi*f4
    a1=1
    a2= 1/(Q4*w4)
    a3=1/(w4**2)
    b1=1
    b2= 1/w3
    return ([b2,b1],[a3,a2,a1])

def EA(pond):
    f5=pond[5]
    f6=pond[7]
    w5=2*np.pi*f5
    w6=2*np.pi*f6
    Q5=pond[6]
    Q6=pond[8]
    a1=w6**2
    a2=w6/Q6
    a3=1
    b1=w5**2
    b2=w5/Q5
    b3=1
    return ([b3,b2,b1],[a3,a2,a1])


def Filtre_flat_h(signal, Fe):
    ONE=Filtre(signal, Fe, Hipass(pflat))
    TWO=Filtre(ONE, Fe, Lopass(pflat))
    Out=TWO
    return (Out)


#def FiltreX(signal, Fe):
#    ONE=Filtre1(signal, Fe, Hipass(w))
#    TWO=Filtre1(ONE, Fe, Lopass(w))
#    THREE=Filtre1(TWO, Fe, TAC(w))
#    FOUR=Filtre1(THREE, Fe, EA(w))
#    Out=FOUR
#    return (Out)




