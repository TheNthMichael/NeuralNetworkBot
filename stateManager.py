import dataEncoder

"""Some things need to be stored globally
for this application to work due to pynput."""

keys_pressed = [0 for x in dataEncoder.KEY_TO_CODE_MAP]

keys_frame_count = [0 for x in dataEncoder.KEY_TO_CODE_MAP]

last_mousex: float = 0
last_mousey: float = 0

last_frame_mousex: float = 0
last_frame_mousey: float = 0

is_recording = False
is_not_exiting = True

screen_cap_scale = 8

monitor_region = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

def update_keys_frame_count():
        assert(len(keys_frame_count) == len(keys_pressed))
        for i in range(len(keys_frame_count)):
            if keys_pressed[i] == 0:
                keys_frame_count[i] = 0
            else:
                keys_frame_count[i] += 1
        return keys_frame_count

def try_add_key_pressed(key):
    try:
        code = dataEncoder.map_key_to_code(key)
        if keys_pressed[code] == 0:
            keys_pressed[code] = 1
    except:
        pass

def try_remove_key_pressed(key):
    try:
        code = dataEncoder.map_key_to_code(key)
        if keys_pressed[code] == 1:
            keys_pressed[code] = 0
    except:
        pass

"""This function is useful for printing the actual names
of the keys pressed and also can be used for clearing all
pressed keys."""
def get_keys_pressed():
    debug_keys_pressed = []
    for i in range(len(keys_pressed)):
        if keys_pressed[i]:
            debug_keys_pressed.append(dataEncoder.map_code_to_key(i))
    #debug_keys_pressed = [dataEncoder.map_code_to_key(x) if keys_pressed[x] else "-" for x in range(len(keys_pressed))].remove("-")
    return debug_keys_pressed

"""This function is useful for printing the actual names
of the keys pressed and also can be used for clearing all
pressed keys."""
def get_keys_pressed(my_keys_pressed):
    debug_keys_pressed = []
    for i in range(len(my_keys_pressed)):
        if my_keys_pressed[i]:
            debug_keys_pressed.append(dataEncoder.map_code_to_key(i))
    #debug_keys_pressed = [dataEncoder.map_code_to_key(x) if keys_pressed[x] else "-" for x in range(len(keys_pressed))].remove("-")
    return debug_keys_pressed

