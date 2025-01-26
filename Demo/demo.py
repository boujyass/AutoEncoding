import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for the grid and label
drawing_grid = np.zeros((28, 28))
current_label = 0
mouse_pressed = False
start_mouse_position = None  # We'll track only the press event position

# Define brush (3x3 example brush)
base_brush = np.array([
    [0,   0.5, 0  ],
    [0.5, 1.0, 0.5],
    [0,   0.5, 0  ]
])
brush_size = base_brush.shape[0]

# Define the Autoencoder class
class Autoencoder(torch.nn.Module):
    def __init__(self, encoder_layers, decoder_layers, activation, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        encoder_modules = []
        input_size = 784
        for layer_size in encoder_layers:
            encoder_modules.append(torch.nn.Linear(input_size, layer_size))
            encoder_modules.append(activation())
            input_size = layer_size
        encoder_modules.append(torch.nn.Linear(input_size, latent_dim))
        self.encoder = torch.nn.Sequential(*encoder_modules)

        # Decoder
        decoder_modules = []
        decoder_input_size = latent_dim + 10
        for layer_size in decoder_layers:
            decoder_modules.append(torch.nn.Linear(decoder_input_size, layer_size))
            decoder_modules.append(activation())
            decoder_input_size = layer_size

        decoder_modules.append(torch.nn.Linear(decoder_input_size, 784))
        decoder_modules.append(torch.nn.Sigmoid())
        self.decoder = torch.nn.Sequential(*decoder_modules)

    def forward(self, x, labels):
        encoded = self.encoder(x)
        labels_onehot = F.one_hot(labels, num_classes=10).float()
        z_cond = torch.cat([encoded, labels_onehot], dim=1)
        decoded = self.decoder(z_cond)
        return encoded, decoded

# Define model configurations
configs = [
    {'name': 'Model1', 'encoder_layers': [500, 100], 'decoder_layers': [100, 500], 'activation': torch.nn.ReLU, 'latent_dim': 2},
    {'name': 'Model2', 'encoder_layers': [500, 250, 100], 'decoder_layers': [100, 250, 500], 'activation': torch.nn.ReLU, 'latent_dim': 2},
    {'name': 'Model3', 'encoder_layers': [500, 100], 'decoder_layers': [100, 500], 'activation': torch.nn.Sigmoid, 'latent_dim': 5},
    {'name': 'Model4', 'encoder_layers': [500, 100], 'decoder_layers': [100, 500], 'activation': torch.nn.Tanh, 'latent_dim': 10},
    {'name': 'Model5', 'encoder_layers': [500, 250, 100], 'decoder_layers': [100, 250, 500], 'activation': torch.nn.ReLU, 'latent_dim': 32},
    {'name': 'Model6', 'encoder_layers': [500, 250, 100], 'decoder_layers': [100, 250, 500], 'activation': torch.nn.ReLU, 'latent_dim': 64},
]

# Initialize and load models
trained_models = {}
for idx, config in enumerate(configs, 1):
    model = Autoencoder(
        encoder_layers=config['encoder_layers'],
        decoder_layers=config['decoder_layers'],
        activation=config['activation'],
        latent_dim=config['latent_dim']
    ).to(device)
    model.load_state_dict(torch.load(f"Model{idx}_model.pth", map_location=device))
    trained_models[config['name']] = model

def reset_grid(event):
    """Reset the drawing grid to all zeros."""
    global drawing_grid
    drawing_grid = np.zeros((28, 28))
    update_plot()

def update_label(text):
    """Update the digit label (0-9)."""
    global current_label
    try:
        current_label = int(text)
        if current_label < 0 or current_label > 9:
            raise ValueError("Label must be between 0 and 9.")
    except ValueError as e:
        print(e)
        current_label = 0

def update_plot():
    """Refresh the plot with the current drawing grid."""
    ax_grid.imshow(drawing_grid, cmap="gray", vmin=0, vmax=1)
    fig.canvas.draw_idle()

def apply_brush(x, y):
    """Apply the defined brush around the point (x, y)."""
    global drawing_grid

    noise = np.random.normal(loc=0, scale=0.1, size=base_brush.shape)
    random_brush = base_brush + noise
    random_brush = np.clip(random_brush, 0, 1)

    half_size = brush_size // 2
    for i in range(-half_size, half_size + 1):
        for j in range(-half_size, half_size + 1):
            xi, yj = x + i, y + j
            if 0 <= xi < 28 and 0 <= yj < 28:
                drawing_grid[yj, xi] = min(1.0, drawing_grid[yj, xi] + random_brush[i + half_size, j + half_size])

def on_press(event):
    """Mouse press event handler - store the starting position."""
    global mouse_pressed, start_mouse_position
    if event.inaxes == ax_grid and event.xdata is not None and event.ydata is not None:
        mouse_pressed = True
        start_mouse_position = (int(event.xdata), int(event.ydata))

def on_release(event):
    """
    Mouse release event handler - store the end position and
    then draw a line (with interpolation) from start to end.
    """
    global mouse_pressed, start_mouse_position
    if mouse_pressed and event.inaxes == ax_grid and event.xdata is not None and event.ydata is not None:
        mouse_pressed = False
        end_mouse_position = (int(event.xdata), int(event.ydata))

        # Interpolate to draw a line
        (x1, y1) = start_mouse_position
        (x2, y2) = end_mouse_position
        steps = max(abs(x2 - x1), abs(y2 - y1))

        if steps == 0:
            # If there's no actual drag (pressed and released in same pixel)
            apply_brush(x1, y1)
        else:
            xs = np.linspace(x1, x2, steps, dtype=int)
            ys = np.linspace(y1, y2, steps, dtype=int)
            for xi, yi in zip(xs, ys):
                apply_brush(xi, yi)

        start_mouse_position = None
        update_plot()

def reconstruct_with_all_models(event):
    """Use each trained model to reconstruct the current drawing."""
    fig_recon, axes = plt.subplots(1, len(trained_models), figsize=(12, 2))
    input_tensor = torch.tensor(drawing_grid, dtype=torch.float32).view(-1, 784).to(device)
    label_tensor = torch.tensor([current_label], dtype=torch.long).to(device)

    for idx, (name, model) in enumerate(trained_models.items()):
        model.eval()
        with torch.no_grad():
            _, reconstruction = model(input_tensor, label_tensor)
        reconstruction_image = reconstruction.view(28, 28).cpu().numpy()
        axes[idx].imshow(reconstruction_image, cmap="gray", vmin=0, vmax=1)
        axes[idx].set_title(name)
        axes[idx].axis("off")

    plt.suptitle("Reconstruction by Models", y=1.05)
    plt.tight_layout()
    plt.show()

# Initialize the figure
fig, ax_grid = plt.subplots(figsize=(6, 6))
ax_grid.imshow(drawing_grid, cmap="gray", vmin=0, vmax=1)
ax_grid.axis("off")

# Connect mouse events
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
# We do NOT connect 'motion_notify_event', so there's no continuous drawing

# Add buttons and text box
ax_reset = plt.axes([0.7, 0.01, 0.1, 0.05])
btn_reset = Button(ax_reset, "Reset")
btn_reset.on_clicked(reset_grid)

ax_reconstruct = plt.axes([0.81, 0.01, 0.15, 0.05])
btn_reconstruct = Button(ax_reconstruct, "Reconstruct")
btn_reconstruct.on_clicked(reconstruct_with_all_models)

ax_label = plt.axes([0.01, 0.01, 0.2, 0.05])
textbox_label = TextBox(ax_label, "Label:")
textbox_label.on_submit(update_label)

plt.show()
