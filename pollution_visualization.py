"""Animate a two-dimensional river pollutant advection-diffusion model."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


@dataclass(frozen=True)
class ModelConfig:
    length_m: float = 2000.0
    width_m: float = 200.0
    nx: int = 101
    ny: int = 31
    depth_m: float = 3.0
    release_mass_mg: float = 10_000_000.0
    release_x_fraction: float = 0.45
    release_y_fraction: float = 0.5
    release_sigma_x_m: float = 45.0
    release_sigma_y_m: float = 14.0
    mean_velocity_mps: float = -0.02
    m2_amplitude_mps: float = 0.06
    s2_amplitude_mps: float = 0.04
    m2_period_hours: float = 12.4206
    s2_period_hours: float = 12.0
    tidal_phase_rad: float = np.pi / 2
    diffusivity_m2ps: float = 0.2
    duration_hours: float = 24.0 * 15.0
    frame_interval_seconds: float = 900.0


@dataclass
class ModelState:
    concentration: np.ndarray
    time_seconds: float = 0.0


def velocity_at(time_seconds: float, config: ModelConfig) -> float:
    m2_period_seconds = config.m2_period_hours * 3600.0
    s2_period_seconds = config.s2_period_hours * 3600.0
    m2_tide = config.m2_amplitude_mps * np.sin(
        2.0 * np.pi * time_seconds / m2_period_seconds + config.tidal_phase_rad
    )
    s2_tide = config.s2_amplitude_mps * np.sin(
        2.0 * np.pi * time_seconds / s2_period_seconds + config.tidal_phase_rad
    )
    return config.mean_velocity_mps + m2_tide + s2_tide


def grid_spacing(config: ModelConfig) -> tuple[float, float]:
    return (
        config.length_m / (config.nx - 1),
        config.width_m / (config.ny - 1),
    )


def stable_time_step(config: ModelConfig) -> float:
    dx, dy = grid_spacing(config)
    max_velocity = (
        abs(config.mean_velocity_mps)
        + config.m2_amplitude_mps
        + config.s2_amplitude_mps
    )
    rate = max_velocity / dx + 2.0 * config.diffusivity_m2ps * (
        1.0 / dx**2 + 1.0 / dy**2
    )
    return 0.4 / rate if rate > 0.0 else 60.0


def create_initial_state(config: ModelConfig) -> ModelState:
    dx, dy = grid_spacing(config)
    x = np.linspace(0.0, config.length_m, config.nx)
    y = np.linspace(0.0, config.width_m, config.ny)
    release_x = config.release_x_fraction * config.length_m
    release_y = config.release_y_fraction * config.width_m
    xx, yy = np.meshgrid(x, y)
    gaussian = np.exp(
        -0.5
        * (
            ((xx - release_x) / config.release_sigma_x_m) ** 2
            + ((yy - release_y) / config.release_sigma_y_m) ** 2
        )
    )
    cell_volume = dx * dy * config.depth_m
    concentration = gaussian * (
        config.release_mass_mg / (gaussian.sum() * cell_volume)
    )
    return ModelState(concentration=concentration)


def advance_one_step(
    state: ModelState, dt: float, config: ModelConfig
) -> ModelState:
    max_step = stable_time_step(config)
    if not np.isfinite(dt) or dt <= 0.0 or dt > max_step * (1.0 + 1e-12):
        raise ValueError("The time step must be positive and satisfy the stability limit.")

    dx, dy = grid_spacing(config)
    concentration = state.concentration
    velocity = velocity_at(state.time_seconds + dt / 2.0, config)

    flux_x = np.empty((config.ny, config.nx + 1), dtype=np.float64)
    flux_x[:, 0] = velocity * concentration[:, 0]
    flux_x[:, -1] = velocity * concentration[:, -1]
    if velocity >= 0.0:
        flux_x[:, 1:-1] = velocity * concentration[:, :-1]
    else:
        flux_x[:, 1:-1] = velocity * concentration[:, 1:]
    advection = -(flux_x[:, 1:] - flux_x[:, :-1]) / dx

    padded = np.pad(concentration, ((1, 1), (1, 1)), mode="edge")
    diffusion = config.diffusivity_m2ps * (
        (padded[1:-1, :-2] - 2.0 * concentration + padded[1:-1, 2:])
        / dx**2
        + (padded[:-2, 1:-1] - 2.0 * concentration + padded[2:, 1:-1])
        / dy**2
    )

    state.concentration = concentration + dt * (advection + diffusion)
    state.time_seconds += dt
    return state


def advance_to(
    state: ModelState, target_time_seconds: float, config: ModelConfig
) -> ModelState:
    if not np.isfinite(target_time_seconds) or target_time_seconds < state.time_seconds:
        raise ValueError("The target time must be finite and no earlier than the current time.")

    max_step = stable_time_step(config)
    while state.time_seconds < target_time_seconds:
        remaining = target_time_seconds - state.time_seconds
        advance_one_step(state, min(max_step, remaining), config)
    return state


def calculate_metrics(state: ModelState, config: ModelConfig) -> dict[str, float]:
    dx, dy = grid_spacing(config)
    x = np.arange(config.nx) * dx
    cell_volume = dx * dy * config.depth_m
    mass = state.concentration.sum() * cell_volume
    weighted_x = (state.concentration * x[np.newaxis, :]).sum() * cell_volume
    return {
        "mass_mg": float(mass),
        "centroid_x_m": float(weighted_x / mass) if mass > 0.0 else 0.0,
        "peak_mg_m3": float(state.concentration.max()),
    }


def main() -> None:
    config = ModelConfig()
    state = create_initial_state(config)
    x = np.linspace(0.0, config.length_m, config.nx)
    initial_peak = calculate_metrics(state, config)["peak_mg_m3"]
    frame_count = int(config.duration_hours * 3600.0 / config.frame_interval_seconds) + 1
    simulation_times = np.linspace(
        0.0, config.duration_hours * 3600.0, frame_count
    )
    tide_curve = np.array(
        [velocity_at(time, config) for time in simulation_times]
    )

    plt.style.use("dark_background")
    fig, (map_axis, profile_axis, tide_axis) = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )
    fig.suptitle(
        "River pollutant dispersion | advection, diffusion & tidal interference",
        fontsize=16,
    )

    image = map_axis.imshow(
        state.concentration,
        origin="lower",
        extent=(0.0, config.length_m, 0.0, config.width_m),
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=initial_peak,
        interpolation="bilinear",
    )
    colorbar = fig.colorbar(image, ax=map_axis, pad=0.02)
    colorbar.set_label("Pollutant concentration (mg/m^3)")
    map_axis.set_xlabel("Distance upstream from river mouth (m)")
    map_axis.set_ylabel("Across-channel distance (m)")
    map_axis.set_title("Concentration field")

    centerline = config.ny // 2
    (profile_line,) = profile_axis.plot(
        x, state.concentration[centerline], color="#58d6c5", linewidth=2
    )
    profile_axis.set_xlim(0.0, config.length_m)
    profile_axis.set_ylim(0.0, initial_peak * 1.1)
    profile_axis.set_xlabel("Distance upstream from river mouth (m)")
    profile_axis.set_ylabel("Centerline (mg/m^3)")
    profile_axis.grid(alpha=0.2)

    tide_axis.plot(
        simulation_times / 3600.0,
        tide_curve,
        color="#f0aa61",
        linewidth=1.5,
        label="M2 + S2 tide",
    )
    tide_axis.axhline(
        config.mean_velocity_mps,
        color="#9aa5b1",
        linestyle="--",
        linewidth=1,
        label="Mean river current",
    )
    (tide_marker,) = tide_axis.plot(
        [0.0], [tide_curve[0]], marker="o", color="#58d6c5", markersize=6
    )
    tide_axis.set_xlim(0.0, config.duration_hours)
    tide_axis.set_xlabel("Simulation time (hours)")
    tide_axis.set_ylabel("Velocity (m/s)")
    tide_axis.set_title("M2 + S2 constituent interference (spring-neap cycle)")
    tide_axis.legend(loc="upper right", fontsize=8)
    tide_axis.grid(alpha=0.2)

    status = map_axis.text(
        0.02,
        0.96,
        "",
        transform=map_axis.transAxes,
        va="top",
        bbox={"facecolor": "#111827", "alpha": 0.85, "edgecolor": "none"},
    )
    fig.text(
        0.5,
        0.005,
        "C_t + u C_x = D (C_xx + C_yy)    |    "
        "u(t) = U + A_M2 sin(2 pi t / T_M2 + phase) "
        "+ A_S2 sin(2 pi t / T_S2 + phase)",
        ha="center",
        fontsize=10,
        color="#b9c3d0",
    )
    fig.tight_layout(rect=(0.03, 0.04, 0.97, 0.95))

    def update(frame: int) -> tuple:
        target_time = min(
            frame * config.frame_interval_seconds,
            config.duration_hours * 3600.0,
        )
        advance_to(state, target_time, config)
        metrics = calculate_metrics(state, config)
        velocity = velocity_at(state.time_seconds, config)
        flow_direction = "upstream" if velocity >= 0.0 else "downstream"
        image.set_data(state.concentration)
        profile_line.set_ydata(state.concentration[centerline])
        tide_marker.set_data(
            [state.time_seconds / 3600.0],
            [velocity_at(state.time_seconds, config)],
        )
        status.set_text(
            f"Time: {state.time_seconds / 3600.0:.2f} h\n"
            f"Flow: {abs(velocity):.3f} m/s {flow_direction}\n"
            f"Plume centroid: {metrics['centroid_x_m']:.0f} m"
        )
        return image, profile_line, tide_marker, status

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=80,
        repeat=False,
        cache_frame_data=False,
    )
    plt.show()
    _ = animation


if __name__ == "__main__":
    main()
