import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import copy, math

class ConstructionLayoutEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Construction Layout Editor")
        self.root.geometry("1000x700")

        # State variables
        self.image = None
        self.photo = None

        # Store markings
        self.highlights = []  # dicts: {'x','y','radius','color'(hex)}
        self.cameras = []     # dicts: {'x','y','radius','fov','angle','color','coverage_color'}
        self.selected_camera = None
        self.selected_highlight = None

        self.mode = tk.StringVar(value="highlight")

        # Drawing settings defaults
        self.highlight_color = "#FFFF00"
        self.highlight_radius = tk.IntVar(value=15)
        self.camera_color = "#FF0000"
        self.coverage_color = (0, 0, 255, 80)
        self.coverage_radius = tk.IntVar(value=200)
        self.coverage_angle = tk.IntVar(value=90)
        self.coverage_direction = tk.IntVar(value=0)

        # Undo/Redo
        self.history = []
        self.history_index = -1

        self.create_menu()
        self.create_toolbar()
        self.create_canvas()
        self.create_status_bar()

        # Trace slider changes
        self.coverage_radius.trace_add('write', self.on_camera_param_change)
        self.coverage_angle.trace_add('write', self.on_camera_param_change)
        self.coverage_direction.trace_add('write', self.on_camera_param_change)
        self.highlight_radius.trace_add('write', self.on_highlight_param_change)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Layout", command=self.load_image)
        file_menu.add_command(label="Save Layout", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def create_toolbar(self):
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load", command=self.load_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save", command=self.save_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Redo", command=self.redo).pack(side=tk.LEFT, padx=2)

        mode_frame = ttk.LabelFrame(toolbar, text="Mode", padding=5)
        mode_frame.pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="Highlight Pillar", variable=self.mode,
                        value="highlight").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Camera View", variable=self.mode,
                        value="camera").pack(side=tk.LEFT)

        # Highlight settings
        ttk.Label(toolbar, text="H Radius").pack(side=tk.LEFT, padx=(20,2))
        ttk.Scale(toolbar, from_=5, to=200, variable=self.highlight_radius, orient=tk.HORIZONTAL).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="H Color", command=self.choose_highlight_color).pack(side=tk.LEFT, padx=2)
        self.hcolor_box = tk.Label(toolbar, background=self.highlight_color, width=2)
        self.hcolor_box.pack(side=tk.LEFT)

        # Camera settings
        ttk.Label(toolbar, text="Cam Radius").pack(side=tk.LEFT, padx=(20,2))
        ttk.Scale(toolbar, from_=50, to=1000, variable=self.coverage_radius, orient=tk.HORIZONTAL).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="FOV").pack(side=tk.LEFT, padx=(20,2))
        ttk.Scale(toolbar, from_=1, to=360, variable=self.coverage_angle, orient=tk.HORIZONTAL).pack(side=tk.LEFT)

        ttk.Label(toolbar, text="Cam Angle").pack(side=tk.LEFT, padx=(20,2))
        ttk.Scale(toolbar, from_=0, to=360, variable=self.coverage_direction, orient=tk.HORIZONTAL).pack(side=tk.LEFT)

        ttk.Button(toolbar, text="Cam Color", command=self.choose_camera_color).pack(side=tk.LEFT, padx=2)
        self.ccolor_box = tk.Label(toolbar, background=self.camera_color, width=2)
        self.ccolor_box.pack(side=tk.LEFT)

    def create_canvas(self):
        self.canvas = tk.Canvas(self.root, cursor="cross", bg="#e0e0e0")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)

    def create_status_bar(self):
        self.status = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def choose_highlight_color(self):
        color = colorchooser.askcolor(title="Highlight Color", initialcolor=self.highlight_color)
        if color[1]:
            self.highlight_color = color[1]
            self.hcolor_box.config(background=self.highlight_color)
            if self.selected_highlight is not None:
                self.selected_highlight['color'] = self.highlight_color
                self.push_history()
                self.display_image()

    def choose_camera_color(self):
        color = colorchooser.askcolor(title="Camera Color", initialcolor=self.camera_color)
        if color[1]:
            self.camera_color = color[1]
            self.ccolor_box.config(background=self.camera_color)
            if self.selected_camera is not None:
                self.selected_camera['color'] = self.camera_color
                self.push_history()
                self.display_image()

    def push_history(self):
        self.history = self.history[:self.history_index+1]
        state = {
            'highlights': copy.deepcopy(self.highlights),
            'cameras': copy.deepcopy(self.cameras)
        }
        self.history.append(state)
        self.history_index += 1

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            state = self.history[self.history_index]
            self.highlights = copy.deepcopy(state['highlights'])
            self.cameras = copy.deepcopy(state['cameras'])
            self.selected_camera = None
            self.selected_highlight = None
            self.status.config(text="Undo")
            self.display_image()

    def redo(self):
        if self.history_index < len(self.history)-1:
            self.history_index += 1
            state = self.history[self.history_index]
            self.highlights = copy.deepcopy(state['highlights'])
            self.cameras = copy.deepcopy(state['cameras'])
            self.selected_camera = None
            self.selected_highlight = None
            self.status.config(text="Redo")
            self.display_image()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if not path:
            return
        self.image = Image.open(path).convert("RGBA")
        self.highlights.clear()
        self.cameras.clear()
        self.selected_camera = None
        self.selected_highlight = None
        self.history.clear()
        self.history_index = -1
        self.push_history()
        self.display_image()
        self.status.config(text="Layout loaded")

    def save_image(self):
        if not self.image:
            messagebox.showwarning("Warning", "Load a layout first.")
            return
        combined = Image.alpha_composite(self.image, self._render_overlay())
        save_to = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG","*.png"), ("JPEG","*.jpg")])
        if save_to:
            combined.convert("RGB").save(save_to)
            self.status.config(text="Layout saved")
            messagebox.showinfo("Saved", f"Saved to {save_to}")

    def display_image(self):
        if not self.image:
            return
        combined = Image.alpha_composite(self.image, self._render_overlay())
        self.photo = ImageTk.PhotoImage(combined)
        self.canvas.config(width=combined.width, height=combined.height)
        self.canvas.delete("all")
        self.canvas.create_image(0,0, image=self.photo, anchor=tk.NW)

    def _render_overlay(self):
        overlay = Image.new("RGBA", self.image.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        # Draw highlights semi-transparent
        for hl in self.highlights:
            x, y, r = hl['x'], hl['y'], hl['radius']
            col_hex = hl['color']
            r_c = int(col_hex[1:3], 16)
            g_c = int(col_hex[3:5], 16)
            b_c = int(col_hex[5:7], 16)
            a = 200
            fill = (r_c, g_c, b_c, a)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=fill)
            if hl is self.selected_highlight:
                # outline selected highlight
                outline_r = r + 2
                draw.ellipse((x-outline_r, y-outline_r, x+outline_r, y+outline_r), outline="white")

        # Draw cameras
        for cam in self.cameras:
            x, y = cam['x'], cam['y']
            rc = 5
            draw.ellipse((x-rc, y-rc, x+rc, y+rc), fill=cam.get('color', self.camera_color))
            R = cam['radius']
            fov = cam['fov']
            angle = cam['angle']
            start = angle - fov/2
            end = angle + fov/2
            draw.pieslice((x-R, y-R, x+R, y+R), start, end, fill=cam.get('coverage_color', self.coverage_color))
            if cam is self.selected_camera:
                outline_r = R + 2
                draw.ellipse((x-outline_r, y-outline_r, x+outline_r, y+outline_r), outline="white")
        return overlay

    def on_click(self, event):
        if not self.image:
            return
        x, y = event.x, event.y
        mode = self.mode.get()
        if mode == "highlight":
            # check if near existing highlight
            found = None
            for hl in self.highlights:
                dx = x - hl['x']
                dy = y - hl['y']
                if math.hypot(dx, dy) <= max(hl['radius'], 10):
                    found = hl
                    break
            if found:
                # select existing
                self.selected_highlight = found
                # update slider/color to match
                self.highlight_radius.set(found['radius'])
                self.highlight_color = found['color']
                self.hcolor_box.config(background=self.highlight_color)
                self.status.config(text=f"Selected highlight at ({found['x']},{found['y']})")
                self.display_image()
            else:
                # new highlight
                r = self.highlight_radius.get()
                hl = {'x': x, 'y': y, 'radius': r, 'color': self.highlight_color}
                self.highlights.append(hl)
                self.selected_highlight = hl
                self.push_history()
                self.status.config(text=f"Highlighted pillar at ({x},{y})")
                self.display_image()
        else:
            # camera mode
            found = None
            for cam in self.cameras:
                dx = x - cam['x']
                dy = y - cam['y']
                if math.hypot(dx, dy) <= 10:
                    found = cam
                    break
            if found:
                self.selected_camera = found
                self.coverage_radius.set(found['radius'])
                self.coverage_angle.set(found['fov'])
                self.coverage_direction.set(found['angle'])
                self.camera_color = found.get('color', self.camera_color)
                self.ccolor_box.config(background=self.camera_color)
                self.status.config(text=f"Selected existing camera at ({found['x']},{found['y']})")
                self.display_image()
            else:
                cam = {
                    'x': x,
                    'y': y,
                    'radius': self.coverage_radius.get(),
                    'fov': self.coverage_angle.get(),
                    'angle': self.coverage_direction.get(),
                    'color': self.camera_color,
                    'coverage_color': self.coverage_color
                }
                self.cameras.append(cam)
                self.selected_camera = cam
                self.push_history()
                self.status.config(text=f"Placed new camera at ({x},{y})")
                self.display_image()

    def on_camera_param_change(self, *args):
        cam = self.selected_camera
        if cam is None:
            return
        cam['radius'] = self.coverage_radius.get()
        cam['fov'] = self.coverage_angle.get()
        cam['angle'] = self.coverage_direction.get()
        self.status.config(
            text=f"Updated camera at ({cam['x']},{cam['y']}) → R={cam['radius']}, FOV={cam['fov']}°, Angle={cam['angle']}°"
        )
        self.display_image()

    def on_highlight_param_change(self, *args):
        hl = self.selected_highlight
        if hl is None:
            return
        hl['radius'] = self.highlight_radius.get()
        self.status.config(
            text=f"Updated highlight at ({hl['x']},{hl['y']}) → Radius={hl['radius']}"
        )
        self.display_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = ConstructionLayoutEditor(root)
    root.mainloop()