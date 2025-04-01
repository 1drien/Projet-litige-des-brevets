import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import json

# Sample feature names
feature_names = [
    "year_filing", "tech_field", "patent_scope", "family_size", "grant_lag", "bwd_cits", "npl_cits", "claims",
    "generality", "originality", "renewal", "quality_index_4", "continuation", "DIV", "CIP", "year_grant",
    "Invalidity", "NPE_filed", "NPE_acquired_pre_grant", "UNIVERSITY", "INDIVIDUAL", "country_JP_PAD",
    "country_US_PAD", "small_applicant", "transfer_pre_grant", "foreign_priority", "gov_int"
]

# Create main window
root = tk.Tk()
root.title("Patent Infringement Predictor")
root.geometry("650x450")
root.resizable(False, False)

# Create canvas
canvas = tk.Canvas(root)
canvas.pack(side="left", fill="both", expand=True)
canvas.create_rectangle(0, 0, 650, 50, fill="lightgray")
canvas.create_text(325, 20, text="Patent Infringement Predictor", fill="black", font=("Arial", 12))

# Frame for inputs
scrollable_frame = ttk.Frame(canvas)
canvas.create_window((0, 50), window=scrollable_frame, anchor="nw")

entry_fields = {}

# Create input fields dynamically
for row, feature in enumerate(feature_names):
    frame = ttk.Frame(scrollable_frame)
    frame.grid(row=row // 3, column=row % 3, padx=10, pady=5, sticky="ew")
    label = ttk.Label(frame, text=feature + ":")
    label.pack(side="left")
    
    if feature in {"continuation", "DIV", "CIP", "Invalidity", "NPE_filed", "NPE_acquired_pre_grant",
                    "UNIVERSITY", "INDIVIDUAL", "country_JP_PAD", "country_US_PAD", "small_applicant",
                    "transfer_pre_grant", "foreign_priority", "gov_int"}:
        var = tk.BooleanVar(value=False)
        checkbox = ttk.Checkbutton(frame, variable=var, command=lambda: reset_result(None))
        checkbox.pack(side="right")
        entry_fields[feature] = var
    else:
        entry = tk.Entry(frame)
        entry.pack(side="right", fill="x")
        entry_fields[feature] = entry

result_label = ttk.Label(scrollable_frame, text="", font=("Arial", 14, "bold"), foreground="blue")
result_label.grid(row=len(feature_names) // 3 + 2, column=0, columnspan=3, pady=10)

risk_label = ttk.Label(scrollable_frame, text="", font=("Arial", 12, "bold"), foreground="red")
risk_label.grid(row=len(feature_names) // 3 + 3, column=0, columnspan=3, pady=10)

error_message = None

# Function to reset result when an input changes
def reset_result(event=None):
    global error_message
    result_label.config(text="")
    risk_label.config(text="")
    if error_message:
        canvas.delete(error_message)
        error_message = None
    if event and isinstance(event.widget, tk.Entry):
        event.widget.config(bg="white")

# Function to validate and submit
def submit():
    global error_message
    empty_fields = False
    
    for feature in feature_names:
        if isinstance(entry_fields[feature], tk.Entry):
            if entry_fields[feature].get().strip() == "":
                entry_fields[feature].config(bg="#FFB6C1")
                empty_fields = True
    
    if empty_fields:
        if error_message:
            canvas.delete(error_message)
        error_message = canvas.create_text(325, 40, text="Please fill all the inputs", fill="red", font=("Arial", 8))
        return
    
    reset_result()
    user_data = {feature: int(entry_fields[feature].get()) if isinstance(entry_fields[feature], tk.Entry)
                 else int(entry_fields[feature].get()) for feature in feature_names}
    
    predicted_class = np.random.choice([0, 1])
    risk = np.random.choice(["High", "Low"]) if predicted_class == 1 else "None"
    
    result_text = "Prediction: Infringement detected!" if predicted_class == 1 else "Prediction: No infringement."
    result_label.config(text=result_text, foreground="red" if predicted_class == 1 else "green")
    risk_label.config(text=f"Risk Level: {risk}", foreground="red" if risk == "High" else "green")

def save_to_logs():
    if result_label.cget("text") == "":
        messagebox.showwarning("Warning", "No prediction to save. Submit first.")
        return
    # Collect user input data, checking for BooleanVar or Entry field
    user_data = {}
    for feature in feature_names:
        if isinstance(entry_fields[feature], tk.BooleanVar):
            user_data[feature] = entry_fields[feature].get()  # Get the value of the BooleanVar (True/False)
        else:
            user_data[feature] = entry_fields[feature].get().strip()  # Get the input text from the Entry field
    # Prepare log entry with user data and result
    log_entry = {
        "inputs": user_data,
        "result": result_label.cget("text"),
        "risk": risk_label.cget("text")
    }
    # Corrected file path and writing the log
    log_file_path = "logs.txt"
    try:
        with open(log_file_path, "a") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")
        messagebox.showinfo("Success", "Result saved to logs.txt")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save to logs: {e}")

# Reset result on input change
for entry in entry_fields.values():
    if isinstance(entry, tk.Entry):
        entry.bind("<Key>", reset_result)

# Buttons
button_frame = ttk.Frame(scrollable_frame)
button_frame.grid(row=len(feature_names) // 3 + 1, column=0, columnspan=3, pady=10)

submit_btn = ttk.Button(button_frame, text="Submit", command=submit)
submit_btn.pack(side="left", padx=10)

save_btn = ttk.Button(button_frame, text="Save to Logs", command=save_to_logs)
save_btn.pack(side="right", padx=10)

root.mainloop()