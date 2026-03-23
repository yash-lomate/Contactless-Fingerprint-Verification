"""
Contactless Fingerprint Verification – Unified Application
===========================================================
Modern CustomTkinter GUI implementing the full CL2CB paper pipeline:
  1. Bézier Surface Modeling  (bezier_surface.py)
  2. Score-Level Fusion       (score_fusion.py)
  3. Legacy CNN Classification
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import os
import time
import threading

# Project modules
import bezier_surface as bsm
import score_fusion as sf

# ── Appearance ────────────────────────────────
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ── Color Palette ─────────────────────────────
ACCENT      = "#6C63FF"
ACCENT_DARK = "#5348CC"
CARD_BG     = "#1E1E2E"
DARK_BG     = "#11111B"
TEXT_PRIM    = "#CDD6F4"
TEXT_SEC     = "#A6ADC8"
SUCCESS     = "#A6E3A1"
DANGER      = "#F38BA8"
WARNING     = "#F9E2AF"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CL2CB - Contactless Fingerprint Verification")
        self.geometry("1200x750")
        self.minsize(1000, 650)

        # Grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # State
        self.selected_file = ""
        self.verify_file_1 = ""
        self.verify_file_2 = ""

        self._build_sidebar()
        self._build_home_frame()
        self._build_bezier_frame()
        self._build_verification_frame()
        self._build_classification_frame()

        self._select_frame("home")

    
    #  SIDEBAR
    
    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color=DARK_BG)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(8, weight=1)

        # Logo
        logo = ctk.CTkLabel(self.sidebar, text=" CL2CB",
                            font=ctk.CTkFont(size=22, weight="bold"),
                            text_color=ACCENT)
        logo.grid(row=0, column=0, padx=20, pady=(25, 5))

        subtitle = ctk.CTkLabel(self.sidebar, text="Fingerprint Verification",
                                font=ctk.CTkFont(size=11), text_color=TEXT_SEC)
        subtitle.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Navigation buttons
        nav_data = [
            ("    Home",           "home",           2),
            ("    Bézier Surface", "bezier",         3),
            ("    Verification",   "verification",   4),
            ("    Classification", "classification", 5),
        ]
        self.nav_buttons = {}
        for text, name, row in nav_data:
            btn = ctk.CTkButton(self.sidebar, text=text, height=40,
                                corner_radius=8,
                                fg_color="transparent", text_color=TEXT_PRIM,
                                hover_color=ACCENT_DARK,
                                anchor="w", font=ctk.CTkFont(size=14),
                                command=lambda n=name: self._select_frame(n))
            btn.grid(row=row, column=0, padx=15, pady=4, sticky="ew")
            self.nav_buttons[name] = btn

        # Theme toggle
        theme_label = ctk.CTkLabel(self.sidebar, text="Theme", text_color=TEXT_SEC,
                                   font=ctk.CTkFont(size=11))
        theme_label.grid(row=9, column=0, padx=20, pady=(10, 0))
        self.theme_menu = ctk.CTkOptionMenu(self.sidebar,
                                            values=["Dark", "Light", "System"],
                                            command=lambda v: ctk.set_appearance_mode(v),
                                            width=160, height=30,
                                            fg_color=CARD_BG, button_color=ACCENT)
        self.theme_menu.grid(row=10, column=0, padx=20, pady=(5, 25))

    def _select_frame(self, name):
        frames = {
            "home":           self.home_frame,
            "bezier":         self.bezier_frame,
            "verification":   self.verif_frame,
            "classification": self.class_frame,
        }
        for key, frm in frames.items():
            if key == name:
                frm.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
            else:
                frm.grid_forget()
        # Highlight active
        for key, btn in self.nav_buttons.items():
            if key == name:
                btn.configure(fg_color=ACCENT, text_color="white")
            else:
                btn.configure(fg_color="transparent", text_color=TEXT_PRIM)

    
    #  HOME
    
    def _build_home_frame(self):
        self.home_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)

        # Hero section
        hero = ctk.CTkFrame(self.home_frame, fg_color=CARD_BG, corner_radius=16)
        hero.pack(fill="x", padx=30, pady=(30, 15))

        ctk.CTkLabel(hero, text="Welcome to CL2CB",
                     font=ctk.CTkFont(size=32, weight="bold"),
                     text_color=TEXT_PRIM).pack(pady=(30, 5))
        ctk.CTkLabel(hero, text="Contactless to Contact-Based Fingerprint Verification",
                     font=ctk.CTkFont(size=14), text_color=ACCENT).pack(pady=(0, 20))

        # Pipeline cards
        cards_frame = ctk.CTkFrame(self.home_frame, fg_color="transparent")
        cards_frame.pack(fill="x", padx=30, pady=10)
        cards_frame.grid_columnconfigure((0, 1, 2), weight=1)

        steps = [
            (" Bézier Surface", "3D parametric surface\nfrom control points →\n2D projection pipeline"),
            (" Dual-Branch", "Siamese CNN embeddings\n+ Minutiae ridge matching\n(crossing-number)"),
            (" Score Fusion", "Weighted combination of\nCNN + minutiae scores →\nMatch / No-match"),
        ]
        for idx, (title, desc) in enumerate(steps):
            card = ctk.CTkFrame(cards_frame, fg_color=CARD_BG, corner_radius=12)
            card.grid(row=0, column=idx, padx=8, pady=8, sticky="nsew")
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=16, weight="bold"),
                         text_color=ACCENT).pack(padx=15, pady=(20, 5))
            ctk.CTkLabel(card, text=desc, font=ctk.CTkFont(size=12),
                         text_color=TEXT_SEC, justify="center").pack(padx=15, pady=(0, 20))

        # About
        about = ctk.CTkFrame(self.home_frame, fg_color=CARD_BG, corner_radius=12)
        about.pack(fill="x", padx=30, pady=15)
        about_text = (
            "This application implements the complete CL2CB research pipeline:\n"
            "Contactless fingerprint acquisition → Bézier surface modeling → "
            "3D-to-2D projection →\nDual-branch feature extraction (CNN + Minutiae) → "
            "Score-level fusion → Verification decision."
        )
        ctk.CTkLabel(about, text=about_text, font=ctk.CTkFont(size=13),
                     text_color=TEXT_SEC, justify="center").pack(padx=20, pady=20)

    
    #  BÉZIER SURFACE TAB
    
    def _build_bezier_frame(self):
        self.bezier_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.bezier_frame.grid_columnconfigure(0, weight=1)
        self.bezier_frame.grid_rowconfigure(2, weight=1)

        # Header
        hdr = ctk.CTkFrame(self.bezier_frame, fg_color=CARD_BG, corner_radius=12)
        hdr.pack(fill="x", padx=30, pady=(20, 10))
        ctk.CTkLabel(hdr, text="    Bézier Surface Modeling",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=TEXT_PRIM).pack(side="left", padx=20, pady=15)
        ctk.CTkButton(hdr, text="Select Fingerprint Image", width=200,
                      fg_color=ACCENT, hover_color=ACCENT_DARK,
                      command=self._bezier_select).pack(side="right", padx=20, pady=15)

        # Image display grid
        self.bezier_display = ctk.CTkFrame(self.bezier_frame, fg_color=CARD_BG, corner_radius=12)
        self.bezier_display.pack(fill="both", expand=True, padx=30, pady=10)
        self.bezier_display.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.bezier_display.grid_rowconfigure(1, weight=1)

        titles = ["Original", "Depth Map", "3D Surface (top-down)", "2D Projection"]
        self.bez_labels = []
        for i, t in enumerate(titles):
            ctk.CTkLabel(self.bezier_display, text=t, font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=ACCENT).grid(row=0, column=i, pady=(10, 2))
            lbl = ctk.CTkLabel(self.bezier_display, text="—", width=220, height=220,
                               fg_color="#2A2A3C", corner_radius=8)
            lbl.grid(row=1, column=i, padx=8, pady=(2, 15), sticky="nsew")
            self.bez_labels.append(lbl)

        # Status
        self.bez_status = ctk.CTkLabel(self.bezier_frame, text="Select an image to begin.",
                                       font=ctk.CTkFont(size=13), text_color=TEXT_SEC)
        self.bez_status.pack(pady=(0, 15))

    def _bezier_select(self):
        path = filedialog.askopenfilename(title="Select Fingerprint",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.bez_status.configure(text="Processing Bézier surface…", text_color=WARNING)
        self.update_idletasks()

        try:
            result = bsm.process_fingerprint(path, grid_m=6, grid_n=6, eval_res=120,
                                             output_size=(256, 256))

            # 1 – Original
            self._set_image(self.bez_labels[0], result['gray'], is_gray=True)

            # 2 – Depth map (colorised)
            depth_vis = (result['depth_map'] * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
            self._set_image(self.bez_labels[1], depth_color)

            # 3 – Surface visualisation (top-down Z intensity)
            surface = result['surface']
            z = surface[:, :, 2]
            z_norm = ((z - z.min()) / (z.max() - z.min() + 1e-8) * 255).astype(np.uint8)
            surface_color = cv2.applyColorMap(z_norm, cv2.COLORMAP_VIRIDIS)
            surface_color = cv2.cvtColor(surface_color, cv2.COLOR_BGR2RGB)
            self._set_image(self.bez_labels[2], surface_color)

            # 4 – 2D projection
            self._set_image(self.bez_labels[3], result['projected_image'], is_gray=True)

            self.bez_status.configure(text=f"✅ Done — {os.path.basename(path)}", text_color=SUCCESS)
        except Exception as e:
            self.bez_status.configure(text=f"❌ Error: {e}", text_color=DANGER)

    
    #  VERIFICATION TAB  (Score-Level Fusion)
    
    def _build_verification_frame(self):
        self.verif_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.verif_frame.grid_columnconfigure(0, weight=1)

        # Header
        hdr = ctk.CTkFrame(self.verif_frame, fg_color=CARD_BG, corner_radius=12)
        hdr.pack(fill="x", padx=30, pady=(20, 10))
        ctk.CTkLabel(hdr, text="🔍  1:1 Verification (Score-Level Fusion)",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=TEXT_PRIM).pack(side="left", padx=20, pady=15)

        # Two-image selector
        sel = ctk.CTkFrame(self.verif_frame, fg_color=CARD_BG, corner_radius=12)
        sel.pack(fill="x", padx=30, pady=10)
        sel.grid_columnconfigure((0, 1), weight=1)

        # Image 1
        f1 = ctk.CTkFrame(sel, fg_color="transparent")
        f1.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        ctk.CTkButton(f1, text="Select Image 1 (Probe)", fg_color=ACCENT,
                      hover_color=ACCENT_DARK,
                      command=lambda: self._verif_select(1)).pack(pady=(0, 8))
        self.verif_img1 = ctk.CTkLabel(f1, text="—", width=200, height=200,
                                       fg_color="#2A2A3C", corner_radius=8)
        self.verif_img1.pack()
        self.verif_lbl1 = ctk.CTkLabel(f1, text="No image", text_color=TEXT_SEC,
                                       font=ctk.CTkFont(size=11))
        self.verif_lbl1.pack(pady=4)

        # Image 2
        f2 = ctk.CTkFrame(sel, fg_color="transparent")
        f2.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        ctk.CTkButton(f2, text="Select Image 2 (Gallery)", fg_color=ACCENT,
                      hover_color=ACCENT_DARK,
                      command=lambda: self._verif_select(2)).pack(pady=(0, 8))
        self.verif_img2 = ctk.CTkLabel(f2, text="—", width=200, height=200,
                                       fg_color="#2A2A3C", corner_radius=8)
        self.verif_img2.pack()
        self.verif_lbl2 = ctk.CTkLabel(f2, text="No image", text_color=TEXT_SEC,
                                       font=ctk.CTkFont(size=11))
        self.verif_lbl2.pack(pady=4)

        # Verify button
        ctk.CTkButton(self.verif_frame, text="▶  Run Verification Pipeline", height=45,
                      fg_color=ACCENT, hover_color=ACCENT_DARK,
                      font=ctk.CTkFont(size=16, weight="bold"),
                      command=self._run_verification).pack(padx=30, pady=10, fill="x")

        # Results card
        self.verif_result = ctk.CTkFrame(self.verif_frame, fg_color=CARD_BG, corner_radius=12)
        self.verif_result.pack(fill="x", padx=30, pady=10)
        self.verif_result_label = ctk.CTkLabel(self.verif_result,
                                               text="Select two images and run the pipeline.",
                                               font=ctk.CTkFont(size=14),
                                               text_color=TEXT_SEC)
        self.verif_result_label.pack(padx=20, pady=20)

    def _verif_select(self, idx):
        path = filedialog.askopenfilename(title=f"Select Image {idx}",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        if idx == 1:
            self.verify_file_1 = path
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self._set_image(self.verif_img1, cv2.resize(gray, (200, 200)), is_gray=True)
            self.verif_lbl1.configure(text=os.path.basename(path))
        else:
            self.verify_file_2 = path
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self._set_image(self.verif_img2, cv2.resize(gray, (200, 200)), is_gray=True)
            self.verif_lbl2.configure(text=os.path.basename(path))

    def _run_verification(self):
        if not self.verify_file_1 or not self.verify_file_2:
            messagebox.showwarning("Missing", "Please select both images first.")
            return
        self.verif_result_label.configure(text="Running dual-branch pipeline…", text_color=WARNING)
        self.update_idletasks()

        def _worker():
            try:
                model_path = os.path.join(os.path.dirname(__file__), 'finger_model.h5')
                result = sf.full_verification(self.verify_file_1, self.verify_file_2,
                                              model_path=model_path)
                match_str = "MATCH" if result['is_match'] else "❌  NO MATCH"
                match_color = SUCCESS if result['is_match'] else DANGER

                text = (
                    f"{match_str}\n\n"
                    f"CNN Similarity Score:       {result['cnn_score']:.4f}\n"
                    f"Minutiae Match Score:       {result['minutiae_score']:.4f}\n"
                    f"Fused Score (α=0.5):        {result['fused_score']:.4f}\n\n"
                    f"Minutiae detected:  Image 1 = {len(result['minutiae_1']['all_minutiae'])}   |   "
                    f"Image 2 = {len(result['minutiae_2']['all_minutiae'])}"
                )
                self.after(0, lambda: self.verif_result_label.configure(text=text, text_color=match_color))
            except Exception as e:
                self.after(0, lambda: self.verif_result_label.configure(
                    text=f"❌ Error: {e}", text_color=DANGER))

        threading.Thread(target=_worker, daemon=True).start()

    
    #  CLASSIFICATION TAB  (Legacy CNN)
    
    def _build_classification_frame(self):
        self.class_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.class_frame.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(self.class_frame, fg_color=CARD_BG, corner_radius=12)
        hdr.pack(fill="x", padx=30, pady=(20, 10))
        ctk.CTkLabel(hdr, text="🧠  CNN Classification (1:N)",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=TEXT_PRIM).pack(side="left", padx=20, pady=15)

        # Controls
        ctrl = ctk.CTkFrame(self.class_frame, fg_color=CARD_BG, corner_radius=12)
        ctrl.pack(fill="x", padx=30, pady=10)
        ctrl.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkButton(ctrl, text="1. Select Image", fg_color=ACCENT,
                      hover_color=ACCENT_DARK,
                      command=self._class_select).grid(row=0, column=0, padx=10, pady=15)
        self.btn_preproc = ctk.CTkButton(ctrl, text="2. Preprocess", fg_color=ACCENT,
                                          hover_color=ACCENT_DARK, state="disabled",
                                          command=self._class_preprocess)
        self.btn_preproc.grid(row=0, column=1, padx=10, pady=15)
        self.btn_predict = ctk.CTkButton(ctrl, text="3. Predict", fg_color=ACCENT,
                                          hover_color=ACCENT_DARK, state="disabled",
                                          command=self._class_predict)
        self.btn_predict.grid(row=0, column=2, padx=10, pady=15)

        # Image display
        disp = ctk.CTkFrame(self.class_frame, fg_color=CARD_BG, corner_radius=12)
        disp.pack(fill="both", expand=True, padx=30, pady=10)
        disp.grid_columnconfigure((0, 1, 2), weight=1)
        disp.grid_rowconfigure(1, weight=1)

        for i, t in enumerate(["Original", "Grayscale", "Threshold"]):
            ctk.CTkLabel(disp, text=t, font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=ACCENT).grid(row=0, column=i, pady=(10, 2))

        self.cls_labels = []
        for i in range(3):
            lbl = ctk.CTkLabel(disp, text="—", width=220, height=220,
                               fg_color="#2A2A3C", corner_radius=8)
            lbl.grid(row=1, column=i, padx=8, pady=(2, 15), sticky="nsew")
            self.cls_labels.append(lbl)

        self.cls_status = ctk.CTkLabel(self.class_frame, text="Select an image to begin.",
                                       font=ctk.CTkFont(size=14), text_color=TEXT_SEC)
        self.cls_status.pack(pady=(0, 15))

    def _class_select(self):
        path = filedialog.askopenfilename(title="Select Image",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.selected_file = path
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (220, 220))
        self._set_image(self.cls_labels[0], img, is_gray=True)
        self.cls_labels[1].configure(image=None, text="—")
        self.cls_labels[2].configure(image=None, text="—")
        self.btn_preproc.configure(state="normal")
        self.btn_predict.configure(state="normal")
        self.cls_status.configure(text=f"Selected: {os.path.basename(path)}", text_color=TEXT_PRIM)

    def _class_preprocess(self):
        if not self.selected_file:
            return
        img = cv2.imread(self.selected_file, 1)
        gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gs = cv2.resize(gs, (220, 220))
        _, thresh = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        self._set_image(self.cls_labels[1], gs, is_gray=True)
        self._set_image(self.cls_labels[2], thresh, is_gray=True)
        self.cls_status.configure(text=" Preprocessing complete.", text_color=SUCCESS)

    def _class_predict(self):
        if not self.selected_file:
            return
        self.cls_status.configure(text=" Loading model & predicting…", text_color=WARNING)
        self.update_idletasks()

        def _worker():
            try:
                from tensorflow.keras.models import load_model
                model_path = os.path.join(os.path.dirname(__file__), 'finger_model.h5')
                model = load_model(model_path, compile=False)

                img = Image.open(self.selected_file).resize((64, 64))
                arr = np.array(img).reshape(1, 64, 64, 3).astype('float32') / 255.0

                pred = model.predict(arr, verbose=0)
                idx = int(np.argmax(pred))
                conf = float(pred[0][idx]) * 100

                users = {0:"User 1", 1:"User 2", 2:"User 3", 3:"User 4", 4:"User 5"}
                user = users.get(idx, "Unknown")

                self.after(0, lambda: self.cls_status.configure(
                    text=f" Prediction: {user}  (confidence {conf:.1f}%)", text_color=SUCCESS))
            except Exception as e:
                self.after(0, lambda: self.cls_status.configure(
                    text=f" Error: {e}", text_color=DANGER))

        threading.Thread(target=_worker, daemon=True).start()

    
    #  UTILITIES
    
    def _set_image(self, label: ctk.CTkLabel, img_array: np.ndarray,
                   is_gray: bool = False, size: tuple = None):
        """Display a numpy image array on a CTkLabel."""
        if is_gray and img_array.ndim == 2:
            pil = Image.fromarray(img_array, mode='L').convert('RGB')
        else:
            pil = Image.fromarray(img_array)
        if size is None:
            w = label.cget("width") or 220
            h = label.cget("height") or 220
            size = (w, h)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=size)
        label.configure(image=ctk_img, text="")
        label.image = ctk_img   # prevent GC


if __name__ == "__main__":
    app = App()
    app.mainloop()
