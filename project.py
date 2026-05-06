import tkinter as tk
from tkinter import filedialog, ttk
from functools import partial

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading, time, sounddevice as sd
import soundfile as sf
from tqdm import tqdm

import pyloudnorm as pyln

import librosa, librosa.display
import numpy as np

import os

#Def
a_data = a_sr = None ##audio data = audio sample rate = none
a_dur = 0.0 ## duration of audio
playing = False ## start application in not playing mode
play_on = play_off = 0.0 ## set player on and off to 0
needle_job = flash_job = None ## set needle job and flash job = none 
needle1 = needle2 = bg_cache = None ##set GUI elements to none
ax1 = ax2 = None ##set graph elements to none
beat_times = None ## set beat times to none
last_beat = -1 ## set last beat to -1 indicating last beat

STEPS = [("Loading",15),("Beats",15),("MFCCs",15),("Spectrum",15),("Loudness LUFS",10),("Spectral Centoid",10),("Zero Crossing Rate",10),("Rendering",10)]
TOTAL = sum(w for _,w in STEPS)

class TkBar(tqdm): ##inhrents function, just changing how its displayed
  def __init__(self, *a, **kw):
    self._bar = kw.pop("tk_bar"); self._var = kw.pop("status_var"); self._root = kw.pop("root") ##bar widget
    super().__init__(*a, **kw)
  def display(self, **_ ):
    pct = (self.n / self.total * 100) if self.total else 0
    self._root.after(0, self._push, pct, self.desc or "") ##main thread of tkinter window (no crashes)
  def _push(self, pct, desc): ##update value of progressbar
    self._bar["value"] = pct 
    self._var.set(f"{desc} ({pct:.0f}%)") ##status text
    self._root.update_idletasks() ##force refresh

#this sets up a function call like tkbar(range(100), tk_bar=bar status_var=var, root=root): etc >> 

#https://deepwiki.com/tqdm/tqdm/3.2-gui-implementations
#https://github.com/tqdm/tqdm/blob/master/tqdm/tk.py

def analyse_track(path):
  global a_data, a_sr ## store global audio (playback later) 
  try:
    with TkBar(total=TOTAL, tk_bar=progress_bar, status_var=status_var, root=root, desc="Starting…") as p: ##here is that function ^
      def step(desc, w): p.set_description(desc); p.display(); p.update(w) ##helper function updates descriptions and amount of progress 

      step("Loading Selected Audio", 0)
      y, sr = librosa.load(path, sr=None); a_data, a_sr = y, sr; p.update(20) #load audio y = audio signal, sr = sample rate of file, sr = none (keep sample rate)
      print(f"Loaded: {len(y)} samples {sr} in HZ")

      #https://librosa.org/doc/latest/generated/librosa.load.html

      step("Beats Are Being Detected", 0)
      tempo, bf = librosa.beat.beat_track(y=y, sr=sr) ## tempo in bpm (not completely accurate)
      bt = librosa.frames_to_time(bf, sr=sr); p.update(20) ##bt = beat time stamps useful later in vizualizing them
      print(f"Tempo: {tempo}, {len(bt)} number of beats")
      #https://librosa.org/doc/main/generated/librosa.beat.beat_track.html
      #https://librosa.org/doc/main/generated/librosa.frames_to_time.html


      step("MFCCs Are Being Detected", 0)
      mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13); p.update(25) # Mel-Freqneuncy Cepstel Coefficents (useful in ML)
      print(f"MFCCs:{mfccs.shape}")
      #https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html

      step("Spectrum Being Detected", 0)
      S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max); p.update(25) ## stft = frequency decomp, convert amp to decibels
      print(f"Spectrogram:{S_db.shape}")
      #https://librosa.org/doc/latest/generated/librosa.amplitude_to_db.html

      step("Loudness LUFS Detected (War is Over)", 0)
      meter = pyln.Meter(sr)
      loudness = meter.integrated_loudness(y) ##LUFS is preceived loudness well... a main messurement of loudness within mastering to spotify, soundcloud etc
      print(f"Loudness: {loudness:.2f} - LUFS")
      #https://pypi.org/project/pyloudnorm/
      #https://github.com/csteinmetz1/pyloudnorm


      step("Spectral Centroid Detected", 0)
      S_C = librosa.feature.spectral_centroid(y=y, sr=sr); p.update(10) ##brightness
      print(f"Spectral Centroid: {S_C.shape}")
      #https://librosa.org/doc/latest/generated/librosa.feature.spectral_centroid.html

      step("Zero Crossing Rate Detected", 0)
      ZCR = librosa.feature.zero_crossing_rate(y) ## how many time does the waveform cross 0 down 0 up 0 down 0... good for detecting noise or interferance
      print(f"Zero Crossing Rate: {ZCR.shape}")
      #https://librosa.org/doc/latest/generated/librosa.feature.zero_crossing_rate.html



      step("Rendering The Graphs :)", 0)
    root.after(0, partial(render, y, sr, tempo, bt, mfccs, S_db, loudness, path, S_C, ZCR)) ##schedules the main rendering of the window 
  except Exception as e:
    def _on_error():
      status_var.set(f"ERROR")
      progress_bar.config(value=0)
    root.after(0, _on_error)
  ##exception handling should not throw errors but this keeps the script safe.

#https://docs.python.org/3/library/tkinter.html
#https://docs.python.org/3/library/tkinter.ttk.html#progressbar


def render(y, sr, tempo, bt, mfccs, S_db, loudness, path, S_C, ZCR):
  #ax1 ax2 are the axes of the graph, needle1 needle2 are the cursors, bg_cache backgrounded / cached resources
  #a_dur is total track length, t time axis, beat_times used for triggering flash, last_beat allows for resetting the loop

  global a_dur, needle1, needle2, bg_cache, ax1, ax2, beat_times, last_beat
  fig.clf()
  a_dur = len(y) / sr
  t = np.linspace(0, a_dur, len(y))
  beat_times = bt; last_beat = -1


  #Waveform: audio amp over time, red lines are beats, matplot render.

  ax1 = fig.add_axes([0, 0.5, 1, 0.5])
  ax1.plot(t, y, color="#4A90D9", linewidth=0.6)
  for b in bt: ax1.axvline(x=b, color="#FF0000", linewidth=0.6, alpha=0.8)
  ax1.set_xlim(0, a_dur); ax1.axis("off")

  #Spectrogram

  ax2 = fig.add_axes([0, 0, 1, 0.5])
  librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis=None, ax=ax2)
  ax2.set_xlim(0, a_dur); ax2.axis("off")

  needle1 = ax1.axvline(x=0, color="#FF0000", linewidth=1.2, alpha=0.95, zorder=10, animated=True)
  needle2 = ax2.axvline(x=0, color="#FF0000", linewidth=1.2, alpha=0.95, zorder=10, animated=True)## animation true enables updates

  #Optamised using blitting by storing as background, only the needle is redrawn graphs are not.

  fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
  canvas.draw()
  bg_cache = canvas.copy_from_bbox(fig.bbox)
  #https://matplotlib.org/stable/users/explain/animations/blitting.html

  #Update GUI variables for tkinter to use.
  loudness_var.set(f"Loudness: {loudness:.2f} LUFS")
  spectural_C_var.set(f"Spectral Centroid: {S_C.mean():.2f}")
  ZCR_var.set(f"Zero Crossing Rate: {ZCR.mean():.2f}")
  play_btn.config(state="normal")
  analyse_btn.config(state="normal")
  progress_bar["value"] = 100
  status_var.set(f"You've Got Mail {float(tempo[0]):.1f} BPM | {len(bt)} beats | {a_dur:.1f}s | {sr} Hz")
#https://github.com/librosa/librosa/tree/main
#https://librosa.org/doc/latest/index.html


def tplay():
  global playing, play_on, play_off, needle_job, last_beat ##into global
  if a_data is None: return ## check if audio is playable avoiding unexcpeted exiting
  if playing:
    sd.stop() ##stops playback
    play_off = min((time.perf_counter() - play_on) + play_off, a_dur) ## elapsed time + offset basic clamping
    playing = False ##set state
    if needle_job: root.after_cancel(needle_job); needle_job = None ## cancels needle job so it stops.
    play_btn.config(text="PLAY") ##update text to PLAY
  else:
    if play_off >= a_dur: play_off = 0.0; last_beat = -1 #reset to beginning and beat track.
    def _play_audio():
      sd.play(a_data[int(play_off*a_sr):], samplerate=a_sr, blocksize=2048) #starts playback form the offeset.
    threading.Thread(target=_play_audio, daemon=True).start() ## thread this to seperate GUI from this process.
    play_on = time.perf_counter() #track progresss
    playing = True ##state update
    play_btn.config(text="STOP") ##back to saying STOP.
    clock() ##starts the clock moving the needle and triggering beat flashing.


def clock():
  global needle_job, playing, play_off, last_beat
  if not playing: return ## stop function if the player is not
  elapsed = (time.perf_counter() - play_on) + play_off ## caulculate elapsed time
  if elapsed >= a_dur: ## clamping time 
    elapsed = a_dur; playing = False; play_off = 0.0; last_beat = -1 ## reset all variables if elapsed time is met
    play_btn.config(text="PLAY"); sd.stop() ## stops audio

  if beat_times is not None: ## check value
    nxt = last_beat + 1 ## nxt (next) is last beat
    if nxt < len(beat_times) and elapsed >= beat_times[nxt]: ## if length of beats and elapsed time is > or = beats_times to next beat
      last_beat = nxt; flash() ## then flash if the beat lines up with the next beat ( or syning the beat to the detected beats so it flashs)

  # Redraw background move the needle to the current time and only the needle elements.
  canvas.restore_region(bg_cache)
  needle1.set_xdata([elapsed, elapsed])
  needle2.set_xdata([elapsed, elapsed])
  ax1.draw_artist(needle1); ax2.draw_artist(needle2)
  canvas.blit(fig.bbox)

  if playing: needle_job = root.after(16, clock) ## call after 16ms (aprox 60 fps) to keep the animation going

def flash():
  global flash_job
  dot.itemconfig(oval, fill="#FF0000") ## changes the fill of the oval
  if flash_job: root.after_cancel(flash_job) # if it exists allready cancel that previous flash job
  def _reset_dot():
    dot.itemconfig(oval, fill="#000000") ## back to black oval
  flash_job = root.after(120, _reset_dot) ## after 120ms _reset_dot is called again so its a pulse

def browse():
  p = filedialog.askopenfilename(filetypes=[("Audio","*.mp3 *.wav *.flac *.ogg *.aiff *.aif *.m4a"),("All","*.*")]) ##check for common audio file exstentions
  if p: file_var.set(p) ## check valid else "" or empty if cancel

def start():
  global playing, play_
  if playing: sd.stop(); playing = False; play_off = 0.0; play_btn.config(text="PLAY") ## stop playback, playing = false and change text config
  path = file_var.get().strip() ## pull file name remove whitespaces
  if not path: status_var.set("Please select a audio file"); return ## if no file name then show message the exit
  play_btn.config(state="disabled"); analyse_btn.config(state="disabled") ## prevent spamming or starting aload of instances
  status_var.set("Starting Crunching Through Audio") ## unprofessional term but shows the status as processing audio
  threading.Thread(target=analyse_track, args=(path,), daemon=True).start() ## starts the thread to process the track, as audio is slow its better to thread this, as GUI would just crash instead deamon = true so it can exit properly if need be.

#Not used
# def export(path, y, sr, tempo, bt, mfccs, S_db, loudness, S_C, ZCR):
#   file = os.path.splitext(os.path.basename(path))[0] + "_export.txt"
  
#   with open(file, "w") as f:
#     f.write("-TRACK ANALYSIS-")
#     f.write(f"File: {path}\n")
#     f.write(f"Sample Rate: {sr} Hz\n")
#     f.write(f"Duration: {len(y)/sr:.2f} seconds\n\n")
#     f.write(f"Tempo: {float(tempo[0]):.2f} BPM\n")
#     f.write(f"Number of Beats: {len(bt)}\n")
#     f.write(f"Beat Times (first 20): {bt[:20]}\n\n")
#     f.write(f"Loudness: {loudness:.2f} LUFS\n\n")
#     f.write(f"Spectral Centroid Mean: {S_C.mean():.4f}\n")
#     f.write(f"Zero Crossing Rate Mean: {ZCR.mean():.4f}\n\n")
#     f.write(f"MFCC Shape: {mfccs.shape}\n")
#     f.write(f"MFCC Mean: {mfccs.mean():.4f}\n\n")
#     f.write(f"Spectrogram Shape: {S_db.shape}\n")
#     f.write(f"Spectrogram Mean: {S_db.mean():.4f}\n\n")

#   print(f"Analysis Exported: {file}")


#TKINTER CODE
root = tk.Tk()
root.title("HGG - MIR Tool")
root.iconphoto(True, tk.PhotoImage(file="icon.png")) ## custom icon
root.configure(bg="#000000")

##styling of tkkwindow
style = ttk.Style(); style.theme_use("clam")

style.configure("TFrame", background="#000000")
style.configure("TLabel", background="#000000", foreground="#FF0000", font=("Terminal",10))
style.configure("TButton", background="#000000", foreground="#FF0000", font=("Terminal",10,"bold"), padding=6)
style.map("TButton", background=[("active","#000000")])

style.configure("TEntry", fieldbackground="#000000", foreground="#FFFFFF", insertcolor="#FFFFFF")
style.configure("P.Horizontal.TProgressbar", troughcolor="#000000", background="#FF0000", thickness=4)

top = ttk.Frame(root, padding=10); top.pack(fill="x")
ttk.Label(top, text="HGG - MIR Tool", font=("Terminal",6,"bold"), foreground="#FF0000").pack(side="bottom", pady=(10,0),padx=0, anchor="w")

file_var = tk.StringVar()
ttk.Entry(top, textvariable=file_var, width=55).pack(side="left", padx=(0,6))
ttk.Button(top, text="Select File", command=browse).pack(side="left", padx=(0,6))

analyse_btn = ttk.Button(top, text="Analyse", command=start); analyse_btn.pack(side="left", padx=(0,6))
play_btn = ttk.Button(top, text="PLAY", command=tplay, state="disabled", width=6); play_btn.pack(side="left", padx=(0,10))

loudness_var = tk.StringVar(value="Loudness: -- LUFS")
ttk.Label(top, textvariable=loudness_var).pack(side="right", padx=10)

spectural_C_var = tk.StringVar(value="Spectral Centroid: --")
ttk.Label(top, textvariable=spectural_C_var).pack(side="bottom", padx=10)

ZCR_var = tk.StringVar(value="Zero Crossing Rate: --")
ttk.Label(top, textvariable=ZCR_var).pack(side="bottom", padx=10)

dot = tk.Canvas(top, width=16, height=16, bg="#000000", highlightthickness=0); dot.pack(side="left")
oval = dot.create_oval(2, 2, 14, 14, fill="#790000", outline="")



status_var = tk.StringVar(value="Select a file")
tk.Label(root, textvariable=status_var, bg="#000000", fg="#FFFFFF", anchor="w", font=("Terminal",9), padx=10, pady=4).pack(fill="x", side="bottom")

progress_bar = ttk.Progressbar(root, style="P.Horizontal.TProgressbar", orient="horizontal", mode="determinate")
progress_bar.pack(fill="x", side="bottom")
fig = plt.Figure(figsize=(5,3), dpi=96, facecolor="#000000")

plot_frame = tk.Frame(root, bg="#000000"); plot_frame.pack(fill="both", expand=True)
canvas = FigureCanvasTkAgg(fig, master=plot_frame); canvas.get_tk_widget().pack(fill="both", expand=True)

root.minsize(900, 200); root.geometry("900x300") ##size of window
root.mainloop()
