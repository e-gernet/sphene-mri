import re
import numpy as np
import nibabel as nib
import tkinter as tk
from tkinter import Text, Label
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Button
import contextlib
import joblib

def load_nifti(filepath) :
    """
    Load a Nifti file and return the object data and image
    """
    try:
        img = nib.load(filepath)
        data = img.get_fdata()
    except:
        print("Aucune image chargé, fin de l'opération")
        exit()
    
    return data, img

def choose_nifti():
    """
    Ask the user for a path to the nifti file
    """
    root = tk.Tk()
    root.withdraw()  # cache la fenêtre principale

    filepath = askopenfilename()

    root.destroy()
    return filepath

def choose_acqp():
    """
    Ask the user for a path to the nifti file
    """
    root = tk.Tk()
    root.withdraw()  # cache la fenêtre principale

    filepath = askopenfilename()

    root.destroy()
    return filepath

def load_acqp(filepath) :
    """
    Load a Nifti file and return the object data and image
    """
    try:
        with open(filepath) as f :
            text  = f.read()
            match = re.search(r'##\$ACQ_echo_time=.*?\n([\s\S]*?)##\$', text).group(1).split()

            values = []
            if match:
                for e in match : 
                    values.append(float(e))
            values = np.array(values)
            print("Here are the real time value of the ",len(values)," echos : ", values)
    except:
        print("Aucun fichier ajouté")
        exit()
    return values

def enter_te(data):
    te_values = None  # variable partagée

    def retrieve_te():
        nonlocal te_values
        user_input = T.get("1.0", "end").strip()
        if user_input:
            try:
                te_values = np.array(
                    [float(x) for x in user_input.replace(",", " ").split()]
                )
                root.destroy()  # ferme la fenêtre après validation
            except:
                print("Invalid TE input")

    root = tk.Tk()
    root.geometry("250x170")

    T = Text(root, height=5, width=52)
    l = Label(root, text="Enter TE values (space or comma separated)")
    b = Button(root, text="Confirm", command=retrieve_te)

    T.pack()
    l.pack()
    b.pack()

    root.mainloop()

    # fallback si rien saisi
    if te_values is None or len(te_values) != data.shape[3]:
        print("TE length mismatch, using default indices")
        n_echos = data.shape[3]
        te_values = np.arange(1, n_echos + 1) * 3.0  # 3, 6, 9, ... ms
        print(f"[TE] Aucun fichier fourni — TE synthétiques : {te_values[0]:.1f} à {te_values[-1]:.1f} ms")

    return te_values


def handle_acqp(data):
    """
    How to retrieve acqp information
    """
    result = {"choice": None}

    def set_choice(value):
        result["choice"] = value
        root.destroy()

    root = tk.Tk()
    # set minimum window size value
    root.minsize(200, 100)
    # set maximum window size value
    root.maxsize(200, 100)
    Button(root, text="Load acqp file", command=lambda: set_choice("load")).pack()
    Button(root, text="Generate TE from Nifti shape", command=lambda: set_choice("generate")).pack()
    Button(root, text="Enter TE manually", command=lambda: set_choice("manual")).pack()

    root.mainloop()
    return result


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()