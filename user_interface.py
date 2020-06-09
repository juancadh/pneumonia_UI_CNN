import warnings
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from keras import backend as K
import tensorflow as tf
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import matplotlib
from tensorflow import keras
import h5py
import cv2
import io
import os
import numpy as np
import pandas as pd
from PIL import ImageTk, Image
from tkinter import filedialog
import tkinter as tk
warnings.filterwarnings("ignore")

tf.get_logger().setLevel('ERROR')

matplotlib.use("TkAgg")
np.random.seed(42)
tf.random.set_seed(42)
# ==================================


class Window(tk.Tk):

    def __init__(self):
        super().__init__()

        def exitWindow():
            self.quit
            self.destroy()

        # Load model
        self.model = keras.models.load_model('./final_model.h5')

        self.title("Pneumonia Detection using CNN")
        self.iconbitmap('./xray.ico')
        self.geometry("800x600")
        self.protocol('WM_DELETE_WINDOW', exitWindow)

        self.px_w_resize = 250
        self.px_h_resize = 250

        lbl_title = tk.Label(self, text="PNEUMONIA DETECTION USING CNN", font="Helvetica 16 bold")
        lbl_title.pack(fill=tk.BOTH, padx=10, pady=10)

        # Open Image Button
        btn_open = tk.Button(self, text="Open Picture", command=self.open_image, width=25)
        btn_open.pack(pady=5, padx=5)

        # Show Model Button
        btn_show_model = tk.Button(self, text="Show Model", command=self.show_model_sum, width=25)
        btn_show_model.pack(pady=5, padx=5)

        self.variable = tk.StringVar(self)
        self.variable.set("conv2d_34")
        self.variable.trace("w", self.update_layer)

        w = tk.OptionMenu(self, self.variable, "conv2d_32", "conv2d_33", "conv2d_34")
        w.pack()

        #btn_layers = tk.Button(self, text="Update", command=update_layer)
        # btn_layers.pack()

        # Exit Button
        # btn_exit = tk.Button(self, text="Exit Program", command=self.quit, width=25)
        # btn_exit.pack(pady=5, padx=5)

        # Set image path selected
        self.lbl_1 = tk.Label(self)
        self.lbl_1.pack(pady=20)

        # Predict image using CNN Model
        self.lbl_3 = tk.Label(self, font="Helvetica 13 bold")
        self.lbl_3.pack(pady=5)

        # Load image into label
        self.lbl_2 = tk.Label(self)
        self.lbl_2.pack(side=tk.LEFT, padx=(20, 0))

        self.lbl_5 = tk.Label(self)
        self.lbl_5.pack(side=tk.RIGHT, padx=(0, 20))

        self.lbl_6 = tk.Label(self)
        self.lbl_6.pack(side=tk.RIGHT, padx=(0, 0))

        self.lbl_4 = tk.Label(self)
        self.lbl_4.pack(side=tk.BOTTOM, padx=20, pady=20)

        # self.fig = plt.figure(figsize=(5, 5))  # , dpi=100
        # self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        # self.canvas.get_tk_widget().pack(side=tk.LEFT, padx=20, expand=False)

    def open_image(self):
        self.filename = filedialog.askopenfilename(initialdir="./sample_imgs", title='Select an image', filetypes=[('image files', ('*.png', '*.jpg', '*.jpeg'))])
        if self.filename:
            # Set image path selected
            self.lbl_1.configure(text=self.filename)
            # Load image into label
            img_1 = ImageTk.PhotoImage(Image.open(self.filename).resize((self.px_w_resize, self.px_h_resize), Image.NEAREST))
            self.lbl_2.configure(image=img_1)
            self.lbl_2.image = img_1
            # Predict image using CNN Model
            prediction = self.predict_one(self.filename, plotIt=True)
            self.lbl_3.configure(text=prediction, font="Helvetica 13 bold")
            # Show layers
            # self.create_layers_plot(self.filename)
            # Heatmap (Gradient Class Activation Map)
            self.gradMAP(self.filename, self.variable.get())  # "conv2d_34"

    # Choose layer
    def update_layer(self, *args):
        print("Value is:" + self.variable.get())
        if self.filename:
            # Load image into label
            img_1 = ImageTk.PhotoImage(Image.open(self.filename).resize((self.px_w_resize, self.px_h_resize), Image.NEAREST))
            self.lbl_2.configure(image=img_1)
            self.lbl_2.image = img_1
            # Predict image using CNN Model
            prediction = self.predict_one(self.filename, plotIt=True)
            self.lbl_3.configure(text=prediction, font="Helvetica 13 bold")
            # Show layers
            # self.create_layers_plot(self.filename)
            # Heatmap (Gradient Class Activation Map)
            self.gradMAP(self.filename, self.variable.get())  # "conv2d_34"

    def prepare_data(self, path, img_size=150):
        img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (img_size, img_size))
        new_array_reshape = new_array.reshape(-1, img_size, img_size, 1)/255.0
        return new_array, new_array_reshape

    def predict_one(self, pathImg, plotIt=True):
        categories = ['NORMAL', 'PNEUMONIA']
        _, imgArray_Rs = self.prepare_data(pathImg, 150)
        prediction = self.model.predict([imgArray_Rs])
        return f"The prediction is {categories[int(round(prediction[0][0]))]} with a probability of {round(float(prediction[0][0]),4)}"

    def show_model_sum(self):
        s = io.StringIO()
        self.model.summary(print_fn=lambda x: s.write(x + '\n'))
        model_summary = s.getvalue()
        s.close()
        popup = tk.Tk()
        popup.title('Model')
        label = tk.Label(popup, text=model_summary)
        label.pack(side="top", fill="x", pady=10)
        # B1 = tk.Button(popup, text="Ok", command=popup.destroy)
        # B1.pack()
        popup.mainloop()

    def pred_img_layer(self, model_tk, img_path):
        imgArray, imgArray_Rs = self.prepare_data(img_path, 150)
        conv1_pred = model_tk.predict([imgArray_Rs])
        conv1_pred = np.squeeze(conv1_pred, axis=0)
        conv1_pred = conv1_pred.reshape(conv1_pred.shape[:2])
        return conv1_pred

    def create_layers_plot(self, img_path):

        n_layers = len(self.model.layers)
        n_cols = np.ceil(n_layers/2)

        fig = plt.figure(figsize=(8, 4), dpi=100, facecolor=(0.85, 0.85, 0.85))
        old_model_name = ""
        model_name = ""
        for l in range(1, n_layers):
            model_t = Sequential()
            for i in range(0, l):
                # Only the convoluted layers
                str_model = self.model.layers[i].name
                if not any(ext in str_model for ext in ['dense', 'flatten', 'dropout']):
                    model_t.add(self.model.layers[i])
                    model_name = self.model.layers[i].name
            if old_model_name != model_name:
                model_t.add(Conv2D(1, (1, 1), name='main_output'))
                img_ly_1 = self.pred_img_layer(model_t, img_path)
                plt.subplot(2, n_cols, l)
                plt.imshow(img_ly_1, cmap='viridis')  # , cmap='viridis', cmap='gray'
                plt.axis('off')
                plt.title(f'{model_name}', fontsize=8)
                old_model_name = model_name

        # Save image plot
        plt.savefig("layers_chart.png", bbox_inches='tight', transparent=True)

        img_1 = ImageTk.PhotoImage(Image.open("layers_chart.png"))  # .resize((300, 200), Image.NEAREST))
        self.lbl_4.configure(image=img_1)
        self.lbl_4.image = img_1

    def gradMAP(self, pathImg, layer_name, intensity=0.5, size_plot=600, export_img=True):
        """ This function creates the Gradient Class Activation Map for a given image input, model and layer"""
        # Code based on: https://medium.com/analytics-vidhya/visualizing-activation-heatmaps-using-tensorflow-5bdba018f759

        # from google.colab.patches import cv2_imshow

        # Load the data and prepare it
        imgArray, imgArray_Rs = self.prepare_data(pathImg, 150)

        # Create the gradients for heatmap
        with tf.GradientTape() as tape:
            iterate = tf.keras.models.Model([self.model.inputs], [self.model.output, self.model.get_layer(layer_name).output])
            model_out, last_conv_layer = iterate(imgArray_Rs)
            class_out = model_out[:, np.argmax(model_out[0])]
            grads = tape.gradient(class_out, last_conv_layer)
            pooled_grads = K.mean(grads, axis=(0, 1, 2))

        # Create heatmap
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap.reshape((heatmap.shape[1], heatmap.shape[2]))

        # Superposing heatmap to the image
        img = cv2.imread(pathImg, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(img.shape[1], img.shape[0], 1)
        heatmap2 = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap2 = cv2.applyColorMap(np.uint8(255*heatmap2), cv2.COLORMAP_JET)
        img = heatmap2 * intensity + img
        img_r = cv2.resize(img, (size_plot, size_plot))
        org_r = cv2.resize(cv2.imread(pathImg, cv2.IMREAD_GRAYSCALE), (size_plot, size_plot))

        # Show images
        # cv2.imshow(org_r)
        # cv2.imshow(img_r)
        plt.matshow(heatmap)
        plt.axis('off')
        plt.margins(0, 0)

        # Export images
        if export_img:
            cv2.imwrite('./heatmap_1.png', img_r)
            plt.savefig('./heatmap_2.png', bbox_inches='tight', transparent=True, pad_inches=0)

        img_1 = ImageTk.PhotoImage(Image.open("./heatmap_1.png").resize((self.px_w_resize, self.px_h_resize), Image.NEAREST))
        self.lbl_5.configure(image=img_1)
        self.lbl_5.image = img_1

        img_2 = ImageTk.PhotoImage(Image.open("./heatmap_2.png").resize((self.px_w_resize, self.px_h_resize), Image.NEAREST))
        self.lbl_6.configure(image=img_2)
        self.lbl_6.image = img_2


if __name__ == "__main__":
    window = Window()
    window.mainloop()
