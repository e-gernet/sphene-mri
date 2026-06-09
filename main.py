import os
import numpy as np
from functions.io import choose_nifti, load_nifti, handle_acqp, choose_acqp, load_acqp, enter_te
from functions.display import display_slice
from functions.utils import compute_mask

def main():
    # 1. Chargement
    nifti_file = choose_nifti()
    data, img = load_nifti(nifti_file)

    # 2. TE
    te_choice = handle_acqp(data)
    if te_choice["choice"] == "load":
        acqp_file = choose_acqp()
        te_values = load_acqp(acqp_file)
    elif te_choice["choice"] == "manual":
        te_values = enter_te(data)
    else:
        n_echos = data.shape[3]
        te_values = np.arange(1, n_echos + 1) * 3.0  # 3, 6, 9, ... ms
        print(f"[TE] Aucun fichier fourni — TE synthétiques : {te_values[0]:.1f} à {te_values[-1]:.1f} ms")

    # 3. Masque
    mask = compute_mask(data, method="rician")

    # 4. Affichage
    display_slice(data, te_values, mask=mask)

if __name__ == "__main__":
    main()

    #Montrer à claude si bien fait fit bi et map pour eff et eff off
    #Demander à claude inversion parametre numpy y x si il a pris en compte
    #Demander à claude pas bouton dedié map lissé à côté d'util pas trop chargé car on enlève t2 eff
    #Demander à claude 3d view pourquoi se base sur mono map, juste une 3d de l'échantillon de base
    #Demander à claude si bien enlever eff de utils, qui devrait être bon car remplacé
    #Demander à claude stocker résultat bouton à l'autre
    #Demander à claude ajustement pour perfection des modèles
    #Demander à claude de repasser sur nos modification
    #signification densité sur histogramme