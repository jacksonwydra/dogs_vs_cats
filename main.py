import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

import torch
from torchvision import transforms
from PIL import ImageTk,Image

from dogs_vs_cats_model import DogsVsCats

def main():
    root = tk.Tk()
    root.title('Dog or Cat')

    # Center the screen
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    app_width, app_height = 400, 300
    x = (screen_width - app_width) // 2
    y = (screen_height - app_height) // 2
    root.geometry(f'{app_width}x{app_height}+{x}+{y}')

    model = DogsVsCats(128)
    model.load_state_dict(torch.load('model2.pt', map_location=torch.device('cpu')))
    model.eval()
    mean = torch.tensor([0.4632, 0.4293, 0.3929])
    std = torch.tensor([0.2646, 0.2550, 0.2536])
    display_transform = transforms.Compose([transforms.Resize(128),
                                            transforms.CenterCrop(128)])

    model_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])

    def select_file():
        filetypes = (
            ("PNG", "*.png"),
            ("JPG", "*.jpg")
        )
        filename = fd.askopenfilename(title='Open a file', initialdir='/', filetypes=filetypes)
        image = display_transform(Image.open(filename))

        img = ImageTk.PhotoImage(image)
        display_img.config(image=img)
        display_img.image = img

        image = model_transform(image)[None, :, :, :]
        label = model(image).argmax().item()
        txt = 'dog' if label else 'cat'
        img_label.config(text=txt)

    open_button = ttk.Button(root, text='Open Picture of Dog or Cat', command=select_file)
    open_button.pack(expand=True)

    display_img = tk.Label()
    display_img.pack(expand=True, pady=20)

    img_label = ttk.Label()
    img_label.pack(expand=True, pady=20)

    root.mainloop()


if __name__ == '__main__':
    main()
