

import pygame
import sounddevice as sd
import numpy as np
import sys
from scipy.fft import fft
import tkinter as tk
from tkinter import ttk, messagebox

# --- Audio Configuration ---
# Default device (used with --no-gui flag)
# Voicemeeter VAIO Input device - using A1 output as input source
# Device 5: Voicemeeter Out A1 (VB-Audio Voicemeeter VAIO), MME (2 in, 0 out)
DEVICE_NAME = 5

# Usage:
#   python sounds.py           - Opens GUI device selector
#   python sounds.py --no-gui  - Uses default device (DEVICE_NAME)
SAMPLE_RATE = 48000  # Hz
BLOCK_SIZE = 1024    # Number of audio frames per block

# --- Pygame Window Configuration ---
WIDTH = 800
HEIGHT = 600
FPS = 60

# --- Colors ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

# --- Global variables to share audio data between threads ---
# We use lists to make them mutable so the callback can change them
audio_level = [0.0]
audio_buffer = [np.zeros(BLOCK_SIZE)]
# Smoothing buffer for more stable visualization
smoothed_fft = [np.zeros(64)]
# Smoothing buffer for bar heights
smoothed_heights = [np.zeros(128)]

def select_audio_device(root=None):
    """
    GUI to select audio input device from a list
    Returns the selected device ID or None if cancelled
    """
    if root is None:
        root = tk.Tk()
        root.title("Select Audio Input Device")
        root.geometry("800x600")
    else:
        # Clear existing widgets if reusing window
        for widget in root.winfo_children():
            widget.destroy()
        root.title("Select Audio Input Device")
    
    # Get all available devices
    devices = sd.query_devices()
    
    # Create main frame
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Title
    title_label = ttk.Label(main_frame, text="Select Audio Input Device", font=("Arial", 16, "bold"))
    title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
    
    # Instructions
    instructions = ttk.Label(main_frame, text="Choose the device you want to capture audio from.\nLook for Voicemeeter devices if you're using Voicemeeter.\nYou can test multiple devices - close the pygame window to return here.", 
                           font=("Arial", 10))
    instructions.grid(row=1, column=0, columnspan=2, pady=(0, 10))
    
    # Create treeview for device list
    columns = ("ID", "Name", "Channels", "Type")
    tree = ttk.Treeview(main_frame, columns=columns, show="headings", height=20)
    
    # Configure columns
    tree.heading("ID", text="ID")
    tree.heading("Name", text="Device Name")
    tree.heading("Channels", text="Channels")
    tree.heading("Type", text="Type")
    
    tree.column("ID", width=50)
    tree.column("Name", width=400)
    tree.column("Channels", width=100)
    tree.column("Type", width=100)
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    
    # Populate device list
    selected_device = [None]  # Use list to make it mutable
    
    for i, device in enumerate(devices):
        # Only show input devices (devices that can capture audio)
        if device['max_input_channels'] > 0:
            device_type = "INPUT"
            if 'voicemeeter' in device['name'].lower():
                device_type = "VOICEMEETER"
            
            # Insert into tree
            item_id = tree.insert("", "end", values=(
                i,
                device['name'],
                f"{device['max_input_channels']} in, {device['max_output_channels']} out",
                device_type
            ))
            
            # Highlight Voicemeeter devices
            if 'voicemeeter' in device['name'].lower():
                tree.set(item_id, "Type", "VOICEMEETER")
    
    # Bind selection event
    def on_select(event):
        selection = tree.selection()
        if selection:
            item = tree.item(selection[0])
            selected_device[0] = int(item['values'][0])
    
    tree.bind("<<TreeviewSelect>>", on_select)
    
    # Place tree and scrollbar
    tree.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    scrollbar.grid(row=2, column=1, sticky=(tk.N, tk.S))
    
    # Configure grid weights
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(2, weight=1)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    
    # Buttons frame
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=3, column=0, columnspan=2, pady=(20, 0))
    
    def on_ok():
        if selected_device[0] is not None:
            root.quit()
        else:
            messagebox.showwarning("No Selection", "Please select a device from the list.")
    
    def on_cancel():
        selected_device[0] = None
        root.quit()
    
    def on_test():
        if selected_device[0] is not None:
            try:
                # Test the selected device
                device_info = sd.query_devices(selected_device[0])
                with sd.InputStream(device=selected_device[0], channels=1, samplerate=48000, blocksize=1024):
                    messagebox.showinfo("Test Successful", f"Device {selected_device[0]} works!\n\n{device_info['name']}")
            except Exception as e:
                messagebox.showerror("Test Failed", f"Device {selected_device[0]} failed:\n\n{str(e)}")
        else:
            messagebox.showwarning("No Selection", "Please select a device to test.")
    
    # Buttons
    ttk.Button(button_frame, text="Test Device", command=on_test).pack(side=tk.LEFT, padx=(0, 10))
    ttk.Button(button_frame, text="Start Visualizer", command=on_ok).pack(side=tk.LEFT, padx=(0, 10))
    ttk.Button(button_frame, text="Exit Program", command=on_cancel).pack(side=tk.LEFT)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start the GUI
    root.mainloop()
    
    # Don't call destroy() - it's already destroyed by mainloop()
    return selected_device[0]

def audio_callback(indata, frames, time, status):
    """
    This function is called by sounddevice for each new audio block.
    """
    if status:
        print(f"Audio status: {status}", file=sys.stderr)
    
    # Calculate the volume (Root Mean Square) and update the global variable
    volume_rms = np.sqrt(np.mean(indata**2))
    
    # We use a simple scaling factor. You may need to adjust this
    # depending on your system's audio levels.
    audio_level[0] = volume_rms * 20 # Amplifier
    
    # Store the audio data for spectrum analysis
    audio_buffer[0] = indata.flatten()
    
    # Debug: print audio data occasionally
    if hasattr(audio_callback, 'debug_counter'):
        audio_callback.debug_counter += 1
    else:
        audio_callback.debug_counter = 0
    
    if audio_callback.debug_counter % 100 == 0:  # Every 100 callbacks
        print(f"Audio callback: level={audio_level[0]:.3f}, buffer_size={len(audio_buffer[0])}, max_val={np.max(np.abs(indata)):.3f}")

def run_audio_visualizer(selected_device):
    """
    Run the audio visualizer with the selected device
    """
    # --- Initialize Pygame ---
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simple Audio Visualizer")
    clock = pygame.time.Clock()
    
    print("Pygame initialized successfully")
    print(f"Window size: {WIDTH}x{HEIGHT}")
    
    # Test if window is actually visible
    pygame.display.flip()
    print("Pygame window should now be visible")
    
    # --- Set up and start the audio stream ---
    try:
        # Get device info for better error messages
        device_info = sd.query_devices(selected_device)
        print(f"Using device {selected_device}: {device_info['name']}")
        print(f"Device channels: {device_info['max_input_channels']} in, {device_info['max_output_channels']} out")
        
        with sd.InputStream(device=selected_device,
                             channels=1, # Mono is fine for volume
                             samplerate=SAMPLE_RATE,
                             blocksize=BLOCK_SIZE,
                             callback=audio_callback):
            
            print(f"Listening to {device_info['name']}... Close the window to try another device.")
            print("Starting pygame main loop...")
            
            # --- Main Game Loop ---
            running = True
            frame_count = 0
            while running:
                # Event handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                # --- Drawing ---
                screen.fill(WHITE)
                
                # Draw a simple test to ensure the window is visible
                font = pygame.font.Font(None, 36)
                text = font.render("Audio Visualizer - Device Working!", True, BLACK)
                screen.blit(text, (50, 50))
                
                # Draw current audio level
                level_text = font.render(f"Audio Level: {audio_level[0]:.3f}", True, BLACK)
                screen.blit(level_text, (50, 100))
                
                # Draw audio buffer info
                buffer_text = font.render(f"Buffer Size: {len(audio_buffer[0])}", True, BLACK)
                screen.blit(buffer_text, (50, 130))
                
                # Draw frequency spectrum
                spectrum_data = audio_buffer[0]
                if len(spectrum_data) > 0 and np.max(np.abs(spectrum_data)) > 0:
                    # Apply FFT to get frequency spectrum
                    fft_data = np.abs(fft(spectrum_data))
                    # Take only the first half (positive frequencies)
                    fft_data = fft_data[:len(fft_data)//2]
                    
                    # Normalize and scale for visualization
                    if np.max(fft_data) > 0:
                        fft_data = fft_data / np.max(fft_data)
                    
                    # Apply smoothing to reduce chaos when audio is low/noisy
                    smoothing_factor = 0.3  # Increased smoothing for smoother visual bars
                    if len(smoothed_fft[0]) == len(fft_data):
                        smoothed_fft[0] = smoothing_factor * smoothed_fft[0] + (1 - smoothing_factor) * fft_data
                    else:
                        smoothed_fft[0] = fft_data.copy()
                    
                    # Use smoothed data for visualization
                    fft_data = smoothed_fft[0]
                    
                    # Draw spectrum bars - make them more visible
                    spectrum_width = WIDTH - 100  # Small margins
                    spectrum_x_start = 50
                    spectrum_height = 200  # Fixed height for spectrum area
                    spectrum_y_start = 200  # Start below the text
                    
                    num_bars = min(128, len(fft_data))  # Fewer bars, wider
                    bar_width_spectrum = max(2, spectrum_width // num_bars)  # Wider bars
                    
                    # Draw spectrum background
                    pygame.draw.rect(screen, (240, 240, 240), 
                                   (spectrum_x_start, spectrum_y_start, spectrum_width, spectrum_height))
                    
                    # Calculate raw bar heights first
                    raw_heights = np.zeros(num_bars)
                    for i in range(num_bars):
                        fft_idx = int(i * len(fft_data) / num_bars)
                        magnitude = fft_data[fft_idx]
                        raw_heights[i] = magnitude * spectrum_height * 0.8
                    
                    # Apply smoothing to bar heights
                    height_smoothing = 0.4  # Smoothing factor for bar heights
                    if len(smoothed_heights[0]) == num_bars:
                        smoothed_heights[0] = height_smoothing * smoothed_heights[0] + (1 - height_smoothing) * raw_heights
                    else:
                        smoothed_heights[0] = raw_heights.copy()
                    
                    # Draw the smoothed bars
                    for i in range(num_bars):
                        # Use smoothed height
                        bar_height_spectrum = int(smoothed_heights[0][i])
                        
                        # Position
                        x = spectrum_x_start + i * bar_width_spectrum
                        y_top = spectrum_y_start + spectrum_height - bar_height_spectrum
                        
                        # Color based on original magnitude (not smoothed)
                        fft_idx = int(i * len(fft_data) / num_bars)
                        magnitude = fft_data[fft_idx]
                        if magnitude > 0.7:
                            color = RED
                        elif magnitude > 0.4:
                            color = YELLOW
                        else:
                            color = GREEN
                        
                        # Draw the spectrum bar
                        if bar_height_spectrum > 0:
                            pygame.draw.rect(screen, color, 
                                           (x, y_top, bar_width_spectrum - 1, bar_height_spectrum))
                    
                    # Draw FFT info
                    fft_text = font.render(f"FFT Max: {np.max(fft_data):.3f}, Bars: {num_bars}, Smoothed", True, BLACK)
                    screen.blit(fft_text, (50, 160))
                else:
                    # No audio data - show test pattern
                    spectrum_width = WIDTH - 100
                    spectrum_x_start = 50
                    spectrum_height = 200
                    spectrum_y_start = 200
                    
                    # Draw spectrum background
                    pygame.draw.rect(screen, (240, 240, 240), 
                                   (spectrum_x_start, spectrum_y_start, spectrum_width, spectrum_height))
                    
                    # Draw "No Audio" message
                    no_audio_text = font.render("No Audio Data Detected", True, RED)
                    screen.blit(no_audio_text, (spectrum_x_start + 50, spectrum_y_start + spectrum_height // 2))
                    
                    # Draw test bars
                    num_bars = 32
                    bar_width = max(2, spectrum_width // num_bars)
                    for i in range(num_bars):
                        x = spectrum_x_start + i * bar_width
                        # Create a test pattern
                        test_height = int(50 * (1 + 0.5 * np.sin(i * 0.2 + frame_count * 0.1)))
                        y_top = spectrum_y_start + spectrum_height - test_height
                        color = (100, 100, 255)  # Blue test bars
                        pygame.draw.rect(screen, color, (x, y_top, bar_width - 1, test_height))
                    
                    # Draw test info
                    test_text = font.render("Test Pattern - Check Audio Device", True, BLACK)
                    screen.blit(test_text, (50, 160))

                # --- Update the display ---
                pygame.display.flip()
                
                # Debug output every 60 frames (about 1 second)
                frame_count += 1
                if frame_count % 60 == 0:
                    print(f"Main loop running... frame {frame_count}, audio level: {audio_level[0]:.3f}")
                
                # Cap the frame rate
                clock.tick(FPS)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nThe selected device may not be available or may be in use by another application.")
        print("Try selecting a different device.")

    finally:
        # --- Clean up ---
        pygame.quit()
        print("Stream and visualizer stopped.")

def main():
    """
    Main function to set up device selection and audio visualization loop
    """
    print("=== Audio Device Tester ===")
    print("This program allows you to test multiple audio devices.")
    print("Close the pygame window to return to device selection.")
    print("Select 'Cancel' in the device selector to exit.\n")
    
    # Create the tkinter root window once
    root = None
    if len(sys.argv) <= 1 or sys.argv[1] != "--no-gui":
        root = tk.Tk()
        root.title("Select Audio Input Device")
        root.geometry("800x600")
    
    try:
        while True:
            # --- Device Selection ---
            # Check if user wants to skip GUI (command line argument)
            if len(sys.argv) > 1 and sys.argv[1] == "--no-gui":
                selected_device = DEVICE_NAME
                print(f"Using default device: {selected_device}")
            else:
                print("Opening device selector...")
                selected_device = select_audio_device(root)
                
                if selected_device is None:
                    print("No device selected. Exiting.")
                    break
                
                print(f"Selected device: {selected_device}")
            
            # Run the audio visualizer
            run_audio_visualizer(selected_device)
            
            # If we get here, the pygame window was closed
            print("\nPygame window closed. Returning to device selection...")
            print("Select another device to test, or click 'Cancel' to exit.\n")
    
    finally:
        # Clean up the tkinter window
        if root is not None:
            root.destroy()

if __name__ == "__main__":
    main()