import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# Primary
TUM_BLUE   = "#0065BD"
WHITE      = "#FFFFFF"
BLACK      = "#1A171B"

# Secondary
TUM_DARK_BLUE_1  = "#003359"
TUM_DARK_BLUE_2  = "#005293"
GRAY_80          = "#57585A"
GRAY_50          = "#9C9D9F"
GRAY_20          = "#D9DADB"

# Accent
TUM_GREEN    = "#A2AD00"
TUM_ORANGE   = "#E37222"
TUM_BEIGE    = "#DAD7CB"
TUM_LIGHT_BLUE_1 = "#64A0C8"
TUM_LIGHT_BLUE_2 = "#98C6EA"

# Colormaps: white → TUM colour (useful for heatmaps)
cmap_blue   = LinearSegmentedColormap.from_list("tum_blue",   [WHITE, TUM_BLUE])
cmap_orange = LinearSegmentedColormap.from_list("tum_orange", [WHITE, TUM_ORANGE])
cmap_green  = LinearSegmentedColormap.from_list("tum_green",  [WHITE, TUM_GREEN])
