"""
Helper functions for working with psychopy
"""
from psychopy import event, visual, core


def is_mouse_pressed(test_mouse: event.Mouse):
    for button in test_mouse.getPressed():
        if button != 0:
            return True
    return False


def get_keypress():
    keys = event.getKeys()
    if keys:
        return keys[0]
    else:
        return None


def shutdown(win: visual.Window):
    win.close()
    core.quit()
