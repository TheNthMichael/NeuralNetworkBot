"""Some things need to be stored globally
for this application to work due to pynput."""

keys_pressed = []

last_mousex: float = 0
last_mousey: float = 0

last_frame_mousex: float = 0
last_frame_mousey: float = 0

is_recording = False
is_not_exiting = False

screen_cap_scale = 8