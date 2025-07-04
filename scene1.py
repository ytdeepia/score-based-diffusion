from manim import *
from customtrace import Trace
import numpy as np
import random
import torch

tex_to_color_map = {
    r"f(x, t)": BLUE_D,
    r"g(t)": RED_D,
}

# Set seeds for reproducibility
seed = 13
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# simple pull toward origin
def drift_center(pos):
    return -0.35 * pos


def drift_center_reverse(pos):
    return 0.05 * pos


def no_drift(pos):
    # No drift, just random diffusion
    return np.array([0, 0])


def drift_left_right(pos):
    # Wind flowing from left to right with vertical oscillations
    # Constant rightward force + oscillating vertical component
    horizontal_force = 0.3  # Constant rightward wind
    vertical_oscillation = 0.15 * np.sin(
        2 * pos[0]
    )  # Oscillating up/down based on x-position

    return np.array([horizontal_force, vertical_oscillation])


def drift_swirl(pos):
    # Swirling motion around the origin
    # Force directed towards the origin with a tangential component
    radius = np.linalg.norm(pos)
    if radius == 0:
        return np.array([0, 0])  # Avoid division by zero

    tangential_force = (
        np.array([-pos[1], pos[0]]) / radius
    )  # Perpendicular to position vector
    inward_force = -0.1 * pos / radius  # Pull towards origin

    return inward_force + tangential_force * 0.2  # Combine forces


def init_positions_disk(num_particles, radius=2.0):
    positions = []
    radius = 2.0
    for i in range(num_particles):
        # Generate random angle and radius
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, radius)

        # Calculate position
        x = r * np.cos(angle)
        y = r * np.sin(angle)

        positions.append(np.array([x, y, 0]))
    return positions


def init_positions_circle(num_particles):
    positions = []

    # Parameters for the two moons
    radius = 2.0
    width = 0.6
    separation = 1.0

    for i in range(num_particles):
        # Decide which moon this particle belongs to
        moon = int(np.random.random() > 0.5)

        # Generate angle along the semicircle
        angle = np.random.uniform(0, np.pi)

        # Add some noise in the radial direction
        r = radius + np.random.uniform(-width / 2, width / 2)

        # Calculate position
        if moon == 0:
            # Upper moon
            x = r * np.cos(angle)
            y = r * np.sin(angle) + separation / 2
        else:
            # Lower moon (flipped)
            x = r * np.cos(angle + np.pi)
            y = r * np.sin(angle + np.pi) - separation / 2

        # Add some noise
        x += np.random.normal(0, 0.1)
        y += np.random.normal(0, 0.1)

        # Constrain to axes bounds
        x = np.clip(x, -3.9, 3.9)
        y = np.clip(y, -2.4, 2.4)

        positions.append(np.array([x, y, 0]))

    return positions


def init_positions_uniform(num_particles, xmin, xmax, ymin, ymax):
    positions = []
    for _ in range(num_particles):
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        z = 0
        positions.append(np.array([x, y, z]))
    return positions


def init_positions_moon(num_particles):
    positions = []
    radius = 2.0
    width = 0.6
    separation = 1.0

    for i in range(num_particles):
        # Generate angle along the semicircle
        angle = np.random.uniform(0, np.pi)

        # Add some noise in the radial direction
        r = radius + np.random.uniform(-width / 2, width / 2)

        # Calculate position
        x = r * np.cos(angle)
        y = r * np.sin(angle) + separation / 2

        # Add some noise
        x += np.random.normal(0, 0.1)
        y += np.random.normal(0, 0.1)

        # Constrain to axes bounds
        x = np.clip(x, -3.9, 3.9)
        y = np.clip(y, -2.4, 2.4)

        positions.append(np.array([x, y, 0]))

    return positions


def init_particles(positions, axes, color=BLUE):
    particles = VGroup()
    for pos in positions:
        dot = Dot(
            point=axes.c2p(pos[0], pos[1]),
            radius=0.04,
            z_index=2,
            color=color,
            fill_opacity=0.9,
        )
        particles.add(dot)
    return particles


class Scene1_1(Scene):
    def construct(self):

        # Animate particles
        self.next_section(skip_animations=False)

        num_particles = 200
        dt = 0.05
        speed = 1

        total_time = 5  # seconds
        steps = int(total_time / dt)
        diffusion = 0.1
        xmin, xmax = -5, 5
        ymin, ymax = -2.5, 2.5

        axes = Axes(
            x_range=[xmin, xmax, 1],
            y_range=[ymin, ymax, 1],
        )

        particle_color = BLUE_C

        # Initialize particles as dots with random positions
        # positions = init_positions_uniform(num_particles, xmin, xmax, ymin, ymax)
        positions = init_positions_disk(num_particles, radius=0.2)
        particles = init_particles(positions, axes, color=particle_color)

        self.add(particles)

        def update_particles(mob, dt, drift_func=drift_left_right, **kwargs):
            dt_particle = dt * speed
            for i, p in enumerate(mob):
                pos = positions[i][:2]
                dW = np.random.normal(scale=np.sqrt(dt_particle), size=2)
                dx = drift_func(pos, **kwargs) * dt_particle + diffusion * dW
                positions[i][:2] += dx
                new_pos = axes.c2p(positions[i][0], positions[i][1])
                p.move_to(new_pos)

        particle1 = particles[5]
        particle1.set_color(YELLOW)
        particle1.set_z_index(3)
        particle2 = particles[42]
        particle2.set_color(RED)
        particle2.set_z_index(3)

        diffusion_equation = (
            MathTex(
                r"\mathrm{d}x = f(x, t) \mathrm{d}t + g(t) \mathrm{d}W",
                font_size=48,
                tex_to_color_map=tex_to_color_map,
            )
            .to_edge(UP, buff=0.5)
            .set_z_index(5)
        )
        eq_rect = SurroundingRectangle(
            diffusion_equation,
            color=WHITE,
            buff=0.1,
            z_index=4,
            fill_opacity=0.8,
            fill_color=BLACK,
        )

        img1 = (
            ImageMobject("img/tumtum.jpg")
            .scale_to_fit_width(2)
            .to_edge(LEFT, buff=0.5)
            .set_z_index(4)
        )
        img1.add(SurroundingRectangle(img1, color=WHITE, buff=0.0))

        img2 = (
            ImageMobject("img/landscape.png")
            .scale_to_fit_width(3)
            .to_edge(RIGHT, buff=0.5)
            .set_z_index(4)
        )
        img2.add(SurroundingRectangle(img2, color=WHITE, buff=0.0))

        line1 = (
            Line(
                particle2.get_center(),
                img1.get_corner(UR),
                color=WHITE,
                stroke_width=2,
            )
            .set_z_index(1)
            .add_updater(
                lambda m: m.put_start_and_end_on(
                    particle2.get_center(), img1.get_corner(DR)
                )
            )
        )

        line2 = (
            Line(
                particle1.get_center(),
                img2.get_corner(UL),
                color=WHITE,
                stroke_width=2,
            )
            .set_z_index(1)
            .add_updater(
                lambda m: m.put_start_and_end_on(
                    particle1.get_center(), img2.get_corner(DL)
                )
            )
        )

        self.add_updater(
            lambda dt: update_particles(particles, dt, drift_center_reverse)
        )

        self.play(
            LaggedStart(Create(eq_rect), Write(diffusion_equation), lag_ratio=0.5),
            run_time=3,
        )

        self.play(LaggedStart(Create(line1), FadeIn(img1)))
        self.play(LaggedStart(Create(line2), FadeIn(img2)))

        self.wait(0.76)

        self.remove_updater(
            lambda dt: update_particles(particles, dt, drift_center_reverse)
        )
        self.add_updater(lambda dt: update_particles(particles, dt, drift_center))
        self.play(
            LaggedStart(Uncreate(line1), FadeOut(img1)),
        )
        self.play(
            LaggedStart(Uncreate(line2), FadeOut(img2)),
        )

        self.wait(2)


if __name__ == "__main__":
    scene = Scene1_1()
    scene.render()
