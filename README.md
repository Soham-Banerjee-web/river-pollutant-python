# River pollutant dispersion (Python)

This is a standalone Python project, separate from the browser visualization.
It animates pollutant transport across a two-dimensional river reach using
advection and diffusion while two tidal constituents interfere.

## Run

From this directory, install the plotting and numerical dependencies and launch
the visualization:

```powershell
python -m pip install -r requirements.txt
python pollution_visualization.py
```

Run the numerical tests with:

```powershell
python -m unittest discover -s tests
```

## Model

The depth-averaged concentration follows

```text
∂C/∂t + u(t) ∂C/∂x = D (∂²C/∂x² + ∂²C/∂y²)
```

The upstream-positive flow combines a mean river current and the principal
semidiurnal M2 and S2 tides:

```text
u(t) = U + A_M2 sin(2πt/T_M2 + φ) + A_S2 sin(2πt/T_S2 + φ)
```

Because M2 (12.4206 hours) and S2 (12 hours) have slightly different periods,
their superposition creates a spring-neap beat about every 14.77 days. The
animation spans 15 days and plots the combined tide beneath the pollutant map,
so the increasing and decreasing tidal range is visible alongside the plume.

Advection uses upwind finite-volume fluxes; diffusion uses centered finite
differences. The banks have zero-gradient boundaries and river ends have
advective outflow. A stable explicit timestep is calculated from the grid,
maximum combined tidal velocity, and diffusivity.

The default horizontal diffusivity (`0.2 m²/s`) is an illustrative value chosen
to make plume spreading visible. This model is for visualization, not a
site-calibrated prediction or a turbulence closure. Model parameters can be
edited in `ModelConfig` in `pollution_visualization.py`.
