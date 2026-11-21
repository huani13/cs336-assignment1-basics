import math

def lr_scheduler(t, a_max, a_min, tw, tc):
    if t<tw:
        return a_max * t/tw
    if t<=tc:
        return a_min + (1+math.cos( (t-tw)/(tc-tw) * math.pi )) / 2 * (a_max-a_min)
    return a_min
