"""Run INSIDE ChimeraX: window pixel positions of atoms under the current camera.

    runscript chimerax_project.py <out.json> name=<atom spec> [name=<atom spec> ...]

Writes {"window": [w, h], name: [[x, y], ...], ...}, x right and y down from the window's
top-left corner. A saved image of the same view is the window scaled by its own size, so
render_panel_b.py maps these onto the image with image_size / window.
"""
import json
import sys

import numpy as np
from chimerax.core.commands import AtomSpecArg

view = session.main_view  # noqa: F821  (ChimeraX provides `session`)
cam = view.camera
w, h = view.window_size
half = np.tan(np.radians(cam.field_of_view) / 2)   # field_of_view is horizontal
out = {'window': [w, h]}
for arg in sys.argv[2:]:
    name, spec = arg.split('=', 1)
    atoms = AtomSpecArg.parse(spec, session)[0].evaluate(session).atoms  # noqa: F821
    p = cam.position.inverse().transform_points(atoms.scene_coords)
    x = w / 2 + p[:, 0] / -p[:, 2] / half * (w / 2)
    y = h / 2 - p[:, 1] / -p[:, 2] / half * (w / 2)
    out[name] = np.column_stack([x, y]).round(2).tolist()
with open(sys.argv[1], 'w') as f:
    json.dump(out, f)
