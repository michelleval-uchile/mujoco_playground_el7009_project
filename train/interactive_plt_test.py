import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
import time

print("=== DIAGNÓSTICO DE GRÁFICOS ===")

# 1. Verificar backend
print(f"1. Backend actual: {matplotlib.get_backend()}")

# 2. Verificar backends disponibles
print(f"2. Backends disponibles: {matplotlib.rcsetup.interactive_bk}")
time.sleep(3)
# 3. Probar Tkinter directamente
try:
    root = tk.Tk()
    root.title("Prueba Tkinter")
    label = tk.Label(root, text="¿Ves esta ventana?")
    label.pack()
    root.update()
    print("3. ✅ Tkinter funciona - deberías ver una ventana pequeña")
    root.after(5000, root.destroy)  # Cerrar después de 2 segundos
except Exception as e:
    print(f"3. ❌ Tkinter NO funciona: {e}")

# 4. Probar matplotlib con diferentes métodos
print("\n4. Probando matplotlib...")
time.sleep(3)
# Método 1: plt.show() normal
plt.figure()
plt.plot([1, 2, 3], [1, 4, 2])
plt.title("Prueba 1 - plt.show()")
plt.show(block=False)
plt.pause(2)
plt.close()
print("   ✅ Prueba 1 completada (¿viste una ventana por 2 segundos?)")

time.sleep(3)
# Método 2: Con backend explícito
plt.switch_backend('TkAgg')
plt.figure()
plt.plot([1, 2, 3], [2, 3, 1])
plt.title("Prueba 2 - TkAgg forzado")
plt.show(block=False)
plt.pause(2)
plt.close()
print("   ✅ Prueba 2 completada")

print("\n=== DIAGNÓSTICO COMPLETADO ===")
print("Responde estas preguntas:")
print("- ¿Viste alguna ventana de Tkinter?")
print("- ¿Viste alguna ventana de matplotlib?")
print("- ¿Hubo algún mensaje de error?")