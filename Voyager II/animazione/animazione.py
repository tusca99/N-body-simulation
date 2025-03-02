import time
import math
import re
import glob
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

t=[]
data_x=[]
data_y=[]
data_z=[]
data_vx=[]
data_vy=[]
data_vz=[]
hv=[]
hp=[]
vswap=[]
uv=[]
tl1 = time.time()

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

for file in sorted(glob.glob("object*.dat") , key=numericalSort):
    #piglio le colonne dai file
    t=np.loadtxt(file, usecols=0)
    data_x.append(np.loadtxt(file, usecols=1))
    data_y.append(np.loadtxt(file, usecols=2))
    data_z.append(np.loadtxt(file, usecols=3))
    data_vx.append(np.loadtxt(file, usecols=4))
    data_vy.append(np.loadtxt(file, usecols=5))
    data_vz.append(np.loadtxt(file, usecols=6))
    hp.append(np.loadtxt(file, usecols=7))
    hv.append(np.loadtxt(file, usecols=8))
    #print(file)
    
for swfile in glob.glob("dvswap.dat"):
    vswap = (np.loadtxt(swfile, usecols=0))
    
for uvfile in glob.glob("Ev.dat"):
    kv = (np.loadtxt(uvfile, usecols=0))
    uv = (np.loadtxt(uvfile, usecols=1))
    kvnasa = (np.loadtxt(uvfile, usecols=2))
    uvnasa = (np.loadtxt(uvfile, usecols=3))
    
    
print("Files caricati in", round(time.time() - tl1, 2), "secondi")

    
    
n = len(data_x) #numero corpi
# colori dei pianeti
colors = ["gold", "forestgreen", "darkorange", "royalblue", "red", "indigo", "sandybrown", "olivedrab", "turquoise",
          "magenta","chartreuse"]
# nomi dei pianeti
names = ["Sole", "Mercurio", "Venere", "Terra", "Marte", "Giove", "Saturno", "Urano", "Nettuno", "Voyager II simulazione", "Voyager II Nasa"]

plots = []
plots_trace = []
red_patch = mpatches.Patch(color=None, label='The red data')
time_legend = plt.legend(handles=[red_patch], loc='upper right',facecolor= 'white')
plt.gca().add_artist(time_legend)
plt.title("Viaggio del Voyager II in 20 anni con 50 correzioni alla velocità, sv=0.003")

EVERY = 2   # Ogni quanti dati tenere, se indeciso lasciare 1
LOW = False # Se lo voglio figo metto False, se voglio vedere velocemente se funzia metto True

if LOW: #22 pollici a 1080p
    FRAMES = 1
    FPS = 30
    DPI = 180
    FIG_SIZE = (21.4, 12)

else: #24.5 pollici 4k
    # dovrebbe fare 60 fps in 4k (o in 1080p, non ricordo)
    FRAMES = 3650      # Numero frames (non è arbitrario ma è il numero di step o meno)
    FPS = 60
    DPI = 180           # Più è alto più è nitido
    FIG_SIZE =(21.4,12)       # Più è alto più la risoluzione aumenta

for i in range(n):
    # serve per la leggenda e per fare il puntino sui pianeti
    h, = plt.plot([], [], marker=".", markersize=5,label=names[i], color=colors[i])
    # serve per fare la scia
    ht, = plt.plot([], [], marker=",", markersize=1, color=colors[i])
    
    plots.append(h)     # tiene traccia del punto sul grafico
    plots_trace.append(ht)      # tiene traccia della scia sul grafico

# impostiamo gli assi del grafico
plt.xlim(-3.5e12, 2e12)
plt.ylim(-6e12, 1e12)
fig = plots[0].get_figure()  # type: plt.Figure
fig.set_size_inches(*FIG_SIZE)
legend = plt.legend(loc='lower right', ncol=2, facecolor= 'white')
ax = plt.gca()
ax.set_facecolor('white')
# la coda la vogliamo lunga 100 dati
trace_max = 40000
# e la facciamo partire un po' dopo credo, non ricordo
trace_start = 1


def animate(i):
    # per ogni pianeta
    for j in range(n):
        # impostiamo la x e la y del puntino che rappresenta il pianeta
        plots[j].set_xdata(data_x[j][i * EVERY])
        plots[j].set_ydata(data_y[j][i * EVERY])
        if j >= trace_start:
            start = max(0, i * EVERY - trace_max)
            # impostiamo la scia prendendo gli ultimi tot dati
            plots_trace[j].set_xdata(data_x[j][start: i * EVERY])
            plots_trace[j].set_ydata(data_y[j][start: i * EVERY])

    # per scrivere i giorni passati sul grafico (si suppone che la prima colonna sia il tempo)
    days = t[i * EVERY] / (3600 * 24)
    years = math.floor(days / 365)
    text = f"Anni: {years:02}  Giorni: {math.floor(days - years * 365):03}"
    time_legend.texts[0].set_text(text)
    # ritorniamo gli oggetti che deve ridisegnare (i puntini e le scie)
    return plots + plots_trace[trace_start:]

# ci serve per sapere l'avanzamento
def progress(i, tot):
    if i % 100 == 0 and i:
        print(f"{i}/{tot}")





t1 = time.time()

ani = FuncAnimation(fig, animate, interval=1000 / FPS, blit=True, repeat=True, frames=FRAMES)
writermp4 = animation.FFMpegWriter(fps=FPS)
if LOW:
    filename = "animationlow.mp4"
else:
    filename = "animationhigh.mp4"    
ani.save(filename, writer=writermp4, progress_callback=progress, dpi=DPI)
print("Animazione prodotta in", round(time.time() - t1, 2), "secondi")


fig1, ax1 = plt.subplots(figsize=(21.4,12))
x = hp[(n-2)]
y = hv[(n-2)]
x2 = hp[n-1]
y2 = hv[n-1]
ax1.set(title='Grafico vs Voyager II 50 vcorr, sv 0.003')
ax1.set(xlabel="distanza dal sole [m]", ylabel="velocità dal sole [m/s]")
ax1.plot(x, y, color ='salmon', label='Voyager II simulato')
ax1.plot(x2, y2,color='darkcyan', label='Voyager II nasa')
ax1.grid(axis='both')
ax1.legend(loc='best', ncol=1, facecolor= 'white')
fig1.savefig('VoyagerIIvelpos.png', dpi = 200)


fig2, ax2 = plt.subplots(figsize=(21.4,12))
y3 = np.sqrt((data_x[n-2]-data_x[n-1])**2 +(data_y[n-2]-data_y[n-1])**2 + (data_z[n-2]-data_z[n-1])**2)/hp[n-1]
x3 = range(len(y3))
ax2.set(title='Grafico err. spazi Voyager II 50 vcorr, sv 0.003')
ax2.set(xlabel="giorno", ylabel="rapporto tra il modulo della differenza e modulo nasa")
ax2.scatter(x3, y3, color ='salmon', label='Voyager II simulato', marker='.', s=1)
ax2.vlines(x=vswap, colors='brown', ls='--', label='correzioni velocità', ymin=min(y3),ymax=max(y3),linewidths=0.5)
ax2.legend(loc='best', ncol=1, facecolor= 'white')
ax2.grid(axis='y')
fig2.savefig('VoyagerIIspazi.png', dpi = 200)


fig3, ax3 = plt.subplots(figsize=(21.4,12))
y4 = (uv + kv)
x4 = range(len(y4))
y5 = (uvnasa + kvnasa)
x5 = range(len(y5))
ax3.set(title='Grafico diff. en. Voyager II 50 vcorr, sv 0.003')
ax3.set(xlabel="giorno", ylabel="differenza energia totale")
ax3.scatter(x4, y4, color ='salmon', label='Votafer II simulato', marker='.', s=1)
ax3.plot(x5, y5,color='darkcyan', label='Voyager II nasa')
ax3.vlines(x=vswap, colors='brown', ls='--', label='correzioni velocità', ymin=min(y4),ymax=max(y4),linewidths=0.5)
ax3.legend(loc='best', ncol=1, facecolor= 'white')
ax3.grid(axis='y')
fig3.savefig('VoyagerIIentot.png', dpi = 200)

# avvia in automatico il file con il lettore di file mp4 predefinito
#import os

#os.startfile(filename)