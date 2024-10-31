import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from model import AttentionUNet
import matplotlib.pyplot as plt

# Define transformation (assuming no resizing is needed)
transform = transforms.Compose([transforms.ToTensor()])

# Load a specific model checkpoint
def load_model(checkpoint_path, encoder_name='resnet34'):
    model = AttentionUNet(encoder_name=encoder_name, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Calculate PSNR and SSIM
def calculate_metrics(output, ground_truth):
    psnr_value = psnr(output, ground_truth, data_range=1.0)
    ssim_value = ssim(output, ground_truth, data_range=1.0)
    return psnr_value.item(), ssim_value.item()

# Main GUI Application Class
class ImageRestorationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Restoration App")
        self.root.geometry("1200x800")
        self.root.configure(bg="#34495e")

        self.model = None
        self.image_files = []
        self.index = 0

        # Initialize UI
        self.setup_ui()

    def setup_ui(self):
        path_frame = tk.Frame(self.root, bg="#34495e")
        path_frame.pack(pady=20)

        self.input_folder = tk.StringVar()
        self.gt_folder = tk.StringVar()
        self.mask_folder = tk.StringVar()

        tk.Label(path_frame, text="Degraded Image Folder:", bg="#34495e", fg="white", font=("Arial", 12)).grid(row=0, column=0, padx=10)
        tk.Entry(path_frame, textvariable=self.input_folder, width=50).grid(row=0, column=1, padx=10)
        tk.Button(path_frame, text="Browse", command=lambda: self.load_folder(self.input_folder), bg="#1abc9c", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=10)

        tk.Label(path_frame, text="Ground Truth Folder (optional):", bg="#34495e", fg="white", font=("Arial", 12)).grid(row=1, column=0, padx=10)
        tk.Entry(path_frame, textvariable=self.gt_folder, width=50).grid(row=1, column=1, padx=10)
        tk.Button(path_frame, text="Browse", command=lambda: self.load_folder(self.gt_folder), bg="#1abc9c", fg="white", font=("Arial", 10, "bold")).grid(row=1, column=2, padx=10)

        tk.Label(path_frame, text="Mask Folder (optional):", bg="#34495e", fg="white", font=("Arial", 12)).grid(row=2, column=0, padx=10)
        tk.Entry(path_frame, textvariable=self.mask_folder, width=50).grid(row=2, column=1, padx=10)
        tk.Button(path_frame, text="Browse", command=lambda: self.load_folder(self.mask_folder), bg="#1abc9c", fg="white", font=("Arial", 10, "bold")).grid(row=2, column=2, padx=10)

        tk.Button(path_frame, text="Load Model", command=self.select_model, bg="#2980b9", fg="white", font=("Arial", 10, "bold")).grid(row=3, column=0, pady=15)
        tk.Button(path_frame, text="Run", command=self.run_inference, bg="#2980b9", fg="white", font=("Arial", 10, "bold")).grid(row=3, column=1, pady=15)

        nav_frame = tk.Frame(self.root, bg="#34495e")
        nav_frame.pack()

        tk.Button(nav_frame, text="<< Previous", command=self.prev_image, bg="#9b59b6", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=20)
        tk.Button(nav_frame, text="Next >>", command=self.next_image, bg="#9b59b6", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=20)

        self.display_frame = tk.Frame(self.root, bg="#34495e")
        self.display_frame.pack(fill="both", expand=True, pady=20)

    def load_folder(self, variable):
        path = filedialog.askdirectory()
        if path:
            variable.set(path)

    def select_model(self):
        checkpoint_path = filedialog.askopenfilename(filetypes=[("Checkpoint files", "*.pth")])
        if checkpoint_path:
            self.model = load_model(checkpoint_path)
            messagebox.showinfo("Model Loaded", "Model loaded successfully.")

    def run_inference(self):
        if not self.model or not self.input_folder.get():
            messagebox.showerror("Error", "Model or input folder not provided.")
            return

        input_folder = self.input_folder.get()
        self.image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        self.index = 0
        self.display_image()

    def display_image(self):
        if not self.image_files:
            return

        filename = self.image_files[self.index]
        degraded_img = Image.open(filename).convert("RGB")
        degraded_img_tensor = transform(degraded_img).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Model inference
        output = self.model(degraded_img_tensor).squeeze().cpu().detach().numpy().transpose(1, 2, 0)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(degraded_img)
        axes[0].set_title("Degraded Image")
        axes[0].axis("off")

        gt_img_path = self.get_related_path(self.gt_folder, filename)
        if gt_img_path:
            gt_img = Image.open(gt_img_path).convert("RGB")
            gt_img_tensor = transform(gt_img).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
            psnr_value, ssim_value_output_gt = calculate_metrics(
                torch.tensor(output).unsqueeze(0).permute(0, 3, 1, 2),
                gt_img_tensor
            )
            _, ssim_value_degraded_gt = calculate_metrics(degraded_img_tensor, gt_img_tensor)
            axes[1].imshow(gt_img)
            axes[1].set_title("Ground Truth")
        else:
            psnr_value, ssim_value_output_gt, ssim_value_degraded_gt = None, None, None
            axes[1].axis("off")

        mask_img_path = self.get_related_path(self.mask_folder, filename)
        if mask_img_path:
            mask_img = Image.open(mask_img_path).convert("L")
            axes[2].imshow(mask_img, cmap='gray')
            axes[2].set_title("Mask")
        else:
            axes[2].axis("off")

        axes[3].imshow(np.clip(output, 0, 1))
        if psnr_value is not None and ssim_value_output_gt is not None and ssim_value_degraded_gt is not None:
            axes[3].set_title(
                f"Output\nPSNR: {psnr_value:.2f} dB\nSSIM (Output vs GT): {ssim_value_output_gt:.4f}\n"
                f"SSIM (Degraded vs GT): {ssim_value_degraded_gt:.4f}"
            )
        else:
            axes[3].set_title("Output")
        axes[3].axis("off")

        for widget in self.display_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def get_related_path(self, folder_var, filename):
        if os.path.isdir(folder_var.get()):
            return os.path.join(folder_var.get(), os.path.basename(filename))
        return None

    def next_image(self):
        self.index = (self.index + 1) % len(self.image_files)
        self.display_image()

    def prev_image(self):
        self.index = (self.index - 1) % len(self.image_files)
        self.display_image()

root = tk.Tk()
app = ImageRestorationApp(root)
root.mainloop()
