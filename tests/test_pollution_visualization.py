import unittest
from dataclasses import replace

import numpy as np

from pollution_visualization import (
    ModelConfig,
    advance_one_step,
    advance_to,
    calculate_metrics,
    create_initial_state,
    stable_time_step,
    velocity_at,
)


class PollutionVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            length_m=400.0,
            width_m=100.0,
            nx=41,
            ny=21,
            depth_m=2.0,
            release_x_fraction=0.5,
            release_sigma_x_m=30.0,
            release_sigma_y_m=12.0,
            mean_velocity_mps=0.0,
            m2_amplitude_mps=0.0,
            s2_amplitude_mps=0.0,
            diffusivity_m2ps=0.5,
        )

    def test_initial_release_has_configured_mass(self):
        state = create_initial_state(self.config)
        metrics = calculate_metrics(state, self.config)

        self.assertAlmostEqual(
            metrics["mass_mg"], self.config.release_mass_mg, places=6
        )
        self.assertGreater(metrics["peak_mg_m3"], 0.0)

    def test_diffusion_spreads_release_without_losing_mass(self):
        state = create_initial_state(self.config)
        initial = calculate_metrics(state, self.config)

        advance_to(state, 20.0, self.config)

        final = calculate_metrics(state, self.config)
        self.assertAlmostEqual(final["mass_mg"], initial["mass_mg"], places=6)
        self.assertLess(final["peak_mg_m3"], initial["peak_mg_m3"])

    def test_positive_velocity_moves_centroid_upstream(self):
        config = replace(
            self.config,
            mean_velocity_mps=0.1,
            diffusivity_m2ps=0.0,
        )
        state = create_initial_state(config)
        initial_centroid = calculate_metrics(state, config)["centroid_x_m"]

        advance_to(state, 60.0, config)

        self.assertGreater(
            calculate_metrics(state, config)["centroid_x_m"], initial_centroid
        )

    def test_m2_and_s2_interfere_to_form_spring_neap_cycle(self):
        config = replace(
            self.config,
            m2_amplitude_mps=0.06,
            s2_amplitude_mps=0.04,
        )
        beat_period_seconds = 1.0 / abs(
            1.0 / (config.m2_period_hours * 3600.0)
            - 1.0 / (config.s2_period_hours * 3600.0)
        )
        spring_tide = abs(velocity_at(0.0, config) - config.mean_velocity_mps)
        neap_tide = abs(
            velocity_at(beat_period_seconds / 2.0, config)
            - config.mean_velocity_mps
        )

        self.assertAlmostEqual(beat_period_seconds / 86400.0, 14.77, delta=0.02)
        self.assertGreater(spring_tide, 2.0 * neap_tide)

    def test_unstable_time_step_is_rejected(self):
        state = create_initial_state(self.config)

        with self.assertRaisesRegex(ValueError, "stability limit"):
            advance_one_step(state, stable_time_step(self.config) * 2.0, self.config)


if __name__ == "__main__":
    unittest.main()
