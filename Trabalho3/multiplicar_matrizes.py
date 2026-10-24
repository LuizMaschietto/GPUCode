
import numpy as np

def main():
    try:
        A = np.loadtxt("A.txt", dtype=np.int64, ndmin=2)
        B = np.loadtxt("B.txt", dtype=np.int64, ndmin=2)
    except FileNotFoundError as e:
        print(f"Arquivo não encontrado: {e.filename}")
        print("Execute gerar_matrizes.py primeiro.")
        return

    if A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1]:
        print("As matrizes devem ser quadradas.")
        return

    if A.shape != B.shape:
        print("As matrizes devem ter o mesmo tamanho.")
        return

    C = A @ B

    np.savetxt("C.txt", C, fmt="%d")

    print(f"Multiplicação concluída: matrizes {A.shape[0]}x{A.shape[1]}")
    print("Resultado salvo em C.txt")

    if A.shape[0] <= 10:
        print("\nMatriz A:")
        print(A)
        print("\nMatriz B:")
        print(B)
        print("\nMatriz C = A x B:")
        print(C)

if __name__ == "__main__":
    main()
