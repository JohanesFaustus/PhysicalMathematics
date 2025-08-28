# Program untuk menghitung nilai PI
def hitung_pi():
    x = 0.5
    u0 = x
    x2 = x * x
    hasil_pi = u0
    coeff = 0.5
    error = 1000.0
    k = 1
    eps = 1e-16

    while error > eps:
        u0 = u0 * x2 * coeff / k
        hasil_pi += u0 / (2 * k + 1)
        coeff += 1.0
        k += 1
        error = abs(u0)  # Menggunakan nilai absolut untuk memastikan error positif

    print(f"Nilai PI = {6.0 * hasil_pi:.16f}, Jumlah suku = {k}")

# Jalankan fungsi
hitung_pi()

