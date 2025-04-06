import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import json
import joblib
from tensorflow.keras.models import load_model

# --- Load the model and scaler ---
model_path = "model_fold_1.h5"
scaler_path = "scaler_fold_1.pkl"
log_path = "logs.txt"

model = load_model(model_path)
scaler = joblib.load(scaler_path)

# --- Feature names ---
feature_names = [
    "year_filing", "tech_field", "patent_scope", "family_size", "grant_lag", "bwd_cits", "npl_cits", "claims", "fwd_cits_5",
    "generality", "originality", "renewal", "quality_index_4", "continuation", "DIV", "CIP", "year_grant",
    "Invalidity", "NPE_filed", "NPE_acquired_pre_grant", "UNIVERSITY", "INDIVIDUAL", "country_JP_PAD",
    "country_US_PAD", "small_applicant", "transfer_pre_grant", "foreign_priority", "gov_int"
]

# --- Create main window ---
root = tk.Tk()
root.title("Patent Infringement Predictor")
root.geometry("800x520")
root.resizable(False, False)

canvas = tk.Canvas(root, height=520, width=650)
canvas.pack(side="left", fill="both", expand=True)
canvas.create_rectangle(0, 0, 650, 50, fill="lightgray")
canvas.create_text(325, 25, text="Patent Infringement Predictor", fill="black", font=("Arial", 12))

scrollable_frame = ttk.Frame(canvas)
canvas.create_window((0, 50), window=scrollable_frame, anchor="nw")

entry_fields = {}

# --- Input fields ---
for row, feature in enumerate(feature_names):
    frame = ttk.Frame(scrollable_frame)
    frame.grid(row=row // 3, column=row % 3, padx=10, pady=5, sticky="ew")
    label = ttk.Label(frame, text=feature + ":")
    label.pack(side="left")

    if feature in {
        "continuation", "DIV", "CIP", "Invalidity", "NPE_filed", "NPE_acquired_pre_grant",
        "UNIVERSITY", "INDIVIDUAL", "country_JP_PAD", "country_US_PAD", "small_applicant",
        "transfer_pre_grant", "foreign_priority", "gov_int"
    }:
        var = tk.BooleanVar(value=False)
        checkbox = ttk.Checkbutton(frame, variable=var)
        checkbox.pack(side="right")
        entry_fields[feature] = var
    else:
        entry = tk.Entry(frame)
        entry.pack(side="right", fill="x")
        entry_fields[feature] = entry

# --- Result labels ---
result_label = ttk.Label(scrollable_frame, text="", font=("Arial", 14, "bold"), foreground="blue")
result_label.grid(row=len(feature_names) // 3 + 2, column=0, columnspan=3, pady=10)

probability_label = ttk.Label(scrollable_frame, text="", font=("Arial", 12), foreground="black")
probability_label.grid(row=len(feature_names) // 3 + 3, column=0, columnspan=3, pady=5)

# --- Reset function ---
def reset_result(event=None):
    result_label.config(text="")
    probability_label.config(text="")
    if event and isinstance(event.widget, tk.Entry):
        event.widget.config(bg="white")

# --- Submit function ---
def submit():
    user_input = []
    empty_fields = False

    for feature in feature_names:
        if isinstance(entry_fields[feature], tk.Entry):
            value = entry_fields[feature].get().strip()
            if value == "":
                entry_fields[feature].config(bg="#FFB6C1")
                empty_fields = True
            try:
                user_input.append(float(value))
            except ValueError:
                messagebox.showerror("Error", f"Invalid input for {feature}.")
                return
        else:
            user_input.append(1 if entry_fields[feature].get() else 0)

    if empty_fields:
        messagebox.showwarning("Warning", "Please fill all inputs.")
        return

    reset_result()

    user_data = np.array(user_input).reshape(1, -1).astype(np.float32)
    user_data_scaled = scaler.transform(user_data)

    predicted_proba = model.predict(user_data_scaled)[0][0]
    predicted_class = int(predicted_proba >= 0.5)
    proba_percent = f"{predicted_proba * 100:.2f}%"

    result_text = "Prediction: Infringement detected!" if predicted_class == 1 else "Prediction: No infringement."
    result_label.config(text=result_text, foreground="red" if predicted_class == 1 else "green")
    probability_label.config(text=f"Probability of litigation: {proba_percent}", foreground="black")

# --- Save to logs ---
def save_to_logs():
    if result_label.cget("text") == "":
        messagebox.showwarning("Warning", "No prediction to save.")
        return

    user_data = {
        feature: entry_fields[feature].get() if isinstance(entry_fields[feature], tk.Entry)
        else entry_fields[feature].get() for feature in feature_names
    }

    log_entry = {
        "inputs": user_data,
        "result": result_label.cget("text"),
        "probability": probability_label.cget("text")
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n\n")

    messagebox.showinfo("Success", "Saved to logs.txt")

# --- Bindings ---
for entry in entry_fields.values():
    if isinstance(entry, tk.Entry):
        entry.bind("<Key>", reset_result)
    else:
        entry.trace("w", lambda *args: reset_result())

# --- Buttons ---
button_frame = ttk.Frame(scrollable_frame)
button_frame.grid(row=len(feature_names) // 3 + 1, column=0, columnspan=3, pady=10)

submit_btn = ttk.Button(button_frame, text="Submit", command=submit)
submit_btn.pack(side="left", padx=10)

save_btn = ttk.Button(button_frame, text="Save to Logs", command=save_to_logs)
save_btn.pack(side="right", padx=10)

root.mainloop()
