
import numpy as np

def main():
    n = int(input("Digite o tamanho das matrizes (N): "))


    # Gera matrizes com valores inteiros entre 0 e 9.
    A = np.random.randint(0, 10, size=(n, n))
    B = np.random.randint(0, 10, size=(n, n))

    np.savetxt("A.txt", A, fmt="%d")
    np.savetxt("B.txt", B, fmt="%d")

    print(f"\nMatrizes {n}x{n} geradas!")
    print("Arquivos salvos: A.txt e B.txt")

if __name__ == "__main__":
    main()
