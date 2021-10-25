# Importation du module MPI
from mpi4py import MPI
import numpy as np
import math
# Pour mesurer le temps passer dans une partie du code
import time
# Partie definition des fonctions ////////////////////////////////
def compute_tensors_int(dim):
    """
    Calcul deux paires de tenseurs qui serviront pour définir deux matrices de rang 1

    :param      dim:  La dimension des matrices (carrées)
    :type       dim:  Entier
    """
    u1 = np.array([(2 * i * i) % 19 - 4 for i in range(1, dim + 1)])
    u2 = np.array([(13 * i) % 11 + 4 for i in range(1, dim + 1)])
    v1 = np.array([(7 * i) % 17 + 2 for i in range(1, dim + 1)])
    v2 = np.array([(3 * i * i + 1) % 13 - 7 for i in range(1, dim + 1)])
    return u1, u2, v1, v2
def init_to_zero(dim):
    """
    fonction pour generer une matrix de zeros de dimension dimxdim
    """
    tmp1 = np.array([0 for i in range(dim)])
    return tmp1, tmp1  
def compute_tensors_real(dim):
    """
    Calcul deux paires de tenseurs qui serviront pour définir deux matrices de rang 1
    :param      dim:  La dimension des matrices (carrées)
    :type       dim:  Entier
    """
    u1 = np.cos(np.array([1.67 * i * math.pi / dim for i in range(dim)]))
    u2 = np.sin(np.array([2.03 * i * math.pi / dim for i in range(dim)]))
    v1 = np.cos(np.array([1.23 * i * i * math.pi / (7.5 * dim) for i in range(dim)]))
    v2 = np.sin(np.array([0.675 * i / (3.1 * dim) for i in range(dim)]))
    return u1, u2, v1, v2
def verif_product(uA, vA, uB, vB, C):
    """
    Vérification du résultat produit matrice matrice grâce à la formule :
    A = u_A.v_A^{T}
    B = u_B.v_B^{T}
    => C= A.B = u_A.<v_A|u_B>v_B^{T} où < | > est un produit scalaire
    :param      uA:   Le vecteur u à gauche du produit tensoriel définissant A
    :type       uA:   Vecteur numpy
    :param      vA:   Le vecteur v à droite du produit tensoriel définissant A
    :type       vA:   Vecteur numpy
    :param      uB:   Le vecteur u à gauche du produit tensoriel définissant B
    :type       uB:   Vecteur numpy
    :param      vB:   Le vecteur v à droite du produit tensoriel définissant B
    :type       vB:   Vecteur numpy
    :param      C:    La matrice résultante du produit matrice-matrice
    :type       C:    Un vecteur deux dimensions de numpy
    """
    vA_dot_uB = np.dot(vA, uB)
    diff_values = np.abs(C - np.tensordot(uA, vA_dot_uB * vB, axes=0))
    max_error = np.argmax(diff_values)
    if diff_values.flat[max_error] > 1.0e-10:
        i = max_error // dim
        j = max_error % dim
        val = uA[i] * vA_dot_uB * vB[j]
        print(f"Erreur numerique : valeur attendue pour C({i},{j}) -> {val}")
        print(f"mais la valeur calculee est : {C[i,j]}")
        raise ValueError("Erreur dans le calcul du produit matrice matrice")
# Duplication du communicateur :
comm = MPI.COMM_WORLD.Dup()
# Interrogation du contexte MPI :
rank = comm.rank
size = comm.size
p = int(np.sqrt(size))
I = rank % p
J = rank // p
# key = comm.rank
colonne = comm.Split(J, I)
ligne = comm.Split(I, J)
# Ouverture d'un fichier nom unique mode simple
fileName = f"sortie{rank:03d}.txt"
print(f"filename 1 : {fileName}")

import sys
import time

dim = 12
if len(sys.argv) > 1:
    dim = int(sys.argv[1])
d = int(dim / p)
uA, vA, uB, vB = compute_tensors_int(dim)


# ============ Selection des proportion=========

## initialisation de l'indice i
i_1 = ligne.rank * d
i_2 = i_1 + d
## initialisation de l'indice j
j_1 = colonne.rank * d
j_2 = j_1 + d

# construction de block A_IJ
uA_i = uA[i_1:i_2]
vA_j = vA[j_1:j_2]
A_ij = np.tensordot(uA_i, vA_j, axes=0)

# construction de block B_IJ
uB_i = uB[i_1:i_2]
vB_j = vB[j_1:j_2]
B_ij = np.tensordot(uB_i, vB_j, axes=0)

# initialisation de block de C
uC, vC = init_to_zero(dim)
uC_i = uC[i_1:i_2]
vC_j = vC[j_1:j_2]
C_ij = np.tensordot(uC_i, vC_j, axes=0)

# Assemblage de A tout entier pour comparaison
A = np.tensordot(uA, vA, axes=0)

file = open(fileName, mode="w")
file.write(f"Rang du processus : {rank}\n")
file.write(f"Nombre de processus : {size}\n")
file.write(f"Les valeurs de I : {I}\n\n")
file.write(f"Les valeurs de J : {J}\n\n")

file.write(f"Matrice A :\n{A}\n\n")
# ////////////////////////
file.write(f"la nouvelle Matrice A1 :\n{A_ij}\n\n")

file.write(f"colonne rank : {colonne.rank}\n\n")
file.write(f"ligne rank : {ligne.rank}\n\n")

B = np.tensordot(uB, vB, axes=0)
file.write(f"Matrice B :\n{B}\n\n")
file.write(f"Le sous Block de B :\n{B_ij}\n\n")
for k in range(p):
    A_ik = colonne.bcast(A_ij, root=k)
    B_kj = ligne.bcast(B_ij, root=k)
    C_ij += A_ik.dot(B_kj)
CC = comm.gather(C_ij, root=0)
file.write(f"Le sous Block de la Matrice C = AXB :\n{C_ij}\n\n")
start = time.time()
C_ij = A_ij.dot(B_ij)
C = A.dot(B)
end = time.time()
file.write(f"Matrice C = AxB :\n{C[J*d:J*d+d, I*d:I*d+d ]}\n")
verif_product(uA_i, vA_j, uB_i, vB_j, C_ij)
file.write("Test passe\n")
elapse_time = max(end - start, 1.0e-14)
file.write(f"Temps CPU produit matrice-matrice : {elapse_time} secondes\n")
file.write(
    f"MFlops pour le produit matrice matrice : {2*(dim**3)/elapse_time/1000000} MFlops\n"
)
file.close()
