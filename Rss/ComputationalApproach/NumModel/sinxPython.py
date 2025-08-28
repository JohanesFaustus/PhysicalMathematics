# Program untuk menghitung sin(x) menggunakan deret Taylor
# Berdasarkan kode Fortran asli

import math

def hitung_sin(sudut_derajat):
    """
    Menghitung nilai sinus dari sebuah sudut dalam derajat
    menggunakan deret Taylor.

    Args:
        sudut_derajat: Sudut dalam derajat yang akan dihitung sinusnya.

    Returns:
        Tuple yang berisi nilai sinus dan jumlah suku yang digunakan.
    """
    pi = 3.1415926535897932384
    
    # Mengubah sudut dari derajat ke radian
    x = sudut_derajat * pi / 180.0
    
    # Inisialisasi variabel
    sin_x = x
    u0 = x
    x2 = x * x
    k = 1
    eps = 1e-16
    error = 1000

    # Menggunakan perulangan while untuk menghitung deret Taylor
    while error > eps:
        # Menghitung suku berikutnya dari deret
        u0 = -u0 * x2 / (2.0 * k) / (2.0 * k + 1.0)
        
        # Menambahkan suku ke dalam hasil sin(x)
        sin_x += u0
        
        # Memperbarui nilai error dengan nilai absolut dari suku terakhir
        error = abs(u0)
        
        # Meningkatkan jumlah suku
        k += 1

    return sin_x, k

# Meminta input dari pengguna
try:
    sudut_input = float(input('Masukkan nilai sudut (derajat): '))
    
    # Memanggil fungsi untuk menghitung sinus
    hasil_sin, jumlah_suku = hitung_sin(sudut_input)
    
    # Menampilkan hasil
    print(f'sin({sudut_input}) derajat = {hasil_sin}')
    print(f'Jumlah suku = {jumlah_suku}')
    print(f'Nilai dari modul math.sin: {math.sin(math.radians(sudut_input))}')

except ValueError:
    print("Input tidak valid. Harap masukkan angka.")