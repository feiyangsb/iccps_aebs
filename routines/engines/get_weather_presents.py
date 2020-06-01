import textwrap
import re
import sys
import os

try:
    sys.path.append(os.environ["CARLA_PYTHON"])
    import carla
except:
    raise Exception('No CARLA module found.')

def getWeatherPresents():
    print('weather presets:\n')
    indent = 4 * ' '
    def wrap(text):
        return '\n'.join(textwrap.wrap(text, initial_indent=indent, subsequent_indent=indent))
    print(wrap(', '.join(x for _, x in findWeatherPresets())) + '.\n')
    
def findWeatherPresets():
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), x) for x in presets]