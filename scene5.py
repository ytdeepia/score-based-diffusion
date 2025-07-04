from manim import *
import numpy as np
import random
import torch
from sklearn.datasets import make_moons

tex_to_color_map = {
    r"f(x, t)": BLUE_D,
    r"g(t)": RED_D,
    r"\Delta t": PURPLE_B,
    r"-\frac{1}{2} \beta(t) x": BLUE_D,
    r"\sqrt{\beta(t)}": RED_D,
}

# Set seeds for reproducibility
seed = 2
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# Define a linear beta schedule
def beta(t, beta_0=0.1, beta_T=0.5, T=1000):
    return beta_0 + (beta_T - beta_0) * (t / T)


def simulate_sde(X0, beta_func, T=1000, dt=1.0, save_steps=[0, 10, 50, 1000]):
    """
    Simulate dx = -0.5*beta(t)x dt + sqrt(beta(t)) dW
    Return a list of 2D arrays with particle positions at selected timesteps.
    """
    positions = []
    X = X0.copy()
    n_steps = int(T / dt)

    for step in range(n_steps + 1):
        t = step * dt
        b = beta_func(t)
        drift = -0.5 * b * X * dt
        diffusion = np.sqrt(b) * np.random.normal(0, np.sqrt(dt), size=X.shape)
        X = X + drift + diffusion

        if step in save_steps:
            positions.append(X.copy())

    return positions  # list of arrays of shape (n_particles, 2)


def make_smiley(n_samples=400, noise=0.0, random_state=42):
    """
    Generate points distributed in a smiley face pattern.
    Returns X (positions) and y (labels: 0=left eye, 1=right eye, 2=smile)
    """
    np.random.seed(random_state)

    # Distribute samples across features
    n_left_eye = n_samples // 4
    n_right_eye = n_samples // 4
    n_smile = n_samples - n_left_eye - n_right_eye

    points = []
    labels = []

    # Left eye (vertical ellipse)
    theta = np.random.uniform(0, 2 * np.pi, n_left_eye)
    r = np.random.uniform(1.5, 2, n_left_eye)
    x_left = -1.2 + 0.05 * np.sqrt(r) * np.cos(theta)
    y_left = 0.5 + 1.5 * np.sqrt(r) * np.sin(theta)

    for i in range(n_left_eye):
        points.append([x_left[i], y_left[i]])
        labels.append(0)

    # Right eye (vertical ellipse)
    theta = np.random.uniform(0, 2 * np.pi, n_right_eye)
    r = np.random.uniform(1.5, 2, n_right_eye)
    x_right = 1.2 + 0.05 * np.sqrt(r) * np.cos(theta)
    y_right = 0.5 + 1.5 * np.sqrt(r) * np.sin(theta)

    for i in range(n_right_eye):
        points.append([x_right[i], y_right[i]])
        labels.append(1)

    # Smile (half moon)
    theta = np.random.uniform(np.pi, 2 * np.pi, n_smile)
    r = np.random.uniform(2.5, 3, n_smile)
    x_smile = r * np.cos(theta)
    y_smile = -5 + 1.2 * r * np.sin(theta)

    for i in range(n_smile):
        points.append([x_smile[i], y_smile[i]])
        labels.append(2)

    X = np.array(points)
    y = np.array(labels)

    # Add noise if specified
    if noise > 0:
        X += np.random.normal(0, noise, X.shape)

    return X, y


class Scene1_5(MovingCameraScene):
    def construct(self):

        markov_brownian = MathTex(
            r"\mathrm{d}x =",
            r"- \frac{1}{2} \beta(t) x",
            r"\mathrm{d}t + ",
            r"\sqrt{\beta(t)}",
            r"\mathrm{d}W",
            font_size=36,
        ).to_edge(UP, buff=0.5)
        markov_brownian[1].set_color(BLUE_D)
        markov_brownian[3].set_color(RED_D)
        self.add(markov_brownian)

        # Show example of diffusion on the two moon distribution
        self.next_section(skip_animations=False)

        n_samples = 400
        noise = 0.0
        time_tracker = ValueTracker(0.0)

        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        X = X - np.mean(X, axis=0)

        # Run simulation and get snapshots
        snapshots = simulate_sde(X, beta)

        # Create one Axes per snapshot
        axes_list = VGroup()
        dot_groups = VGroup()

        for _ in range(4):
            ax = Axes(
                x_range=[-3, 3],
                y_range=[-3, 3],
                axis_config={"color": WHITE},
                tips=False,
            ).scale_to_fit_width(3)
            axes_list.add(ax)

        axes_list.arrange(RIGHT, buff=0.25).move_to(0.5 * UP)

        title_0 = Tex(r"Initial distribution", font_size=40).next_to(
            axes_list[0], DOWN, buff=1.0
        )
        title_1 = MathTex(r"t=10", font_size=40).next_to(axes_list[1], DOWN, buff=1.0)
        title_2 = MathTex(r"t=50", font_size=40).next_to(axes_list[2], DOWN, buff=1.0)
        title_3 = MathTex(r"t=1000", font_size=40).next_to(axes_list[3], DOWN, buff=1.0)

        for i, X_snapshot in enumerate(snapshots):
            ax = axes_list[i]
            dots = VGroup()
            for j, pos in enumerate(X_snapshot):
                color = TEAL_B if y[j] == 0 else ORANGE
                dots.add(Dot(point=ax.c2p(pos[0], pos[1]), color=color, radius=0.02))

            dot_groups.add(dots)

        self.play(LaggedStart(FadeIn(dot_groups[0]), Write(title_0), lag_ratio=0.5))

        dots_copy = dot_groups[0].copy()
        self.add(dots_copy)
        self.play(
            LaggedStart(
                ReplacementTransform(dots_copy, dot_groups[1]),
                Write(title_1),
                lag_ratio=0.6,
            ),
            run_time=3,
        )

        dots_copy = dot_groups[1].copy()
        self.add(dots_copy)
        self.play(
            LaggedStart(
                ReplacementTransform(dots_copy, dot_groups[2]),
                Write(title_2),
                lag_ratio=0.6,
            ),
            run_time=3,
        )

        dots_copy = dot_groups[2].copy()
        self.add(dots_copy)
        self.play(
            LaggedStart(
                ReplacementTransform(dots_copy, dot_groups[3]),
                Write(title_3),
                lag_ratio=0.6,
            ),
            run_time=3,
        )
        self.wait(1.4)
        self.play(Circumscribe(markov_brownian[1], color=BLUE_D, run_time=1.5))
        self.wait(0.8)
        self.play(Circumscribe(markov_brownian[3], color=RED_D, run_time=1.5))

        self.play(
            LaggedStart(
                FadeOut(dot_groups[3]),
                FadeOut(dot_groups[2]),
                FadeOut(dot_groups[1]),
                lag_ratio=0.5,
            ),
            run_time=3,
        )

        # Now show the same animation for smiley distribution
        X_smiley, y_smiley = make_smiley(
            n_samples=n_samples, noise=0, random_state=seed
        )
        X_smiley = X_smiley - np.mean(X_smiley, axis=0)

        # Run simulation and get snapshots for smiley
        snapshots_smiley = simulate_sde(X_smiley, beta)

        # Create one Axes per snapshot for smiley
        dot_groups_smiley = VGroup()

        for i, X_snapshot in enumerate(snapshots_smiley):
            ax = axes_list[i]
            dots = VGroup()
            for j, pos in enumerate(X_snapshot):
                if y_smiley[j] == 0:  # left eye
                    color = BLUE
                elif y_smiley[j] == 1:  # right eye
                    color = RED
                else:  # smile
                    color = GREEN
                dots.add(Dot(point=ax.c2p(pos[0], pos[1]), color=color, radius=0.02))

            dot_groups_smiley.add(dots)

        self.play(ReplacementTransform(dot_groups[0], dot_groups_smiley[0]))

        dots_copy = dot_groups_smiley[0].copy()
        self.add(dots_copy)
        self.play(ReplacementTransform(dots_copy, dot_groups_smiley[1]), run_time=2)

        dots_copy = dot_groups_smiley[1].copy()
        self.add(dots_copy)
        self.play(ReplacementTransform(dots_copy, dot_groups_smiley[2]), run_time=2)
        dots_copy = dot_groups_smiley[2].copy()

        self.add(dots_copy)
        self.play(ReplacementTransform(dots_copy, dot_groups_smiley[3]), run_time=2)

        # Show how img act as high dimensional particles
        self.next_section(skip_animations=False)

        ffhq_0 = ImageMobject(
            "img/ffhq_3.png",
        ).scale_to_fit_width(2)
        ffhq_0.add(SurroundingRectangle(ffhq_0, color=WHITE, stroke_width=2, buff=0))
        ffhq_0.next_to(axes_list[0], DR, buff=-1)

        ffhq_1 = ImageMobject(
            "img/ffhq_3_noise_61.png",
        ).scale_to_fit_width(2)
        ffhq_1.add(SurroundingRectangle(ffhq_1, color=WHITE, stroke_width=2, buff=0))
        ffhq_1.next_to(axes_list[1], DR, buff=-0.3)

        ffhq_2 = ImageMobject(
            "img/ffhq_3_noise_198.png",
        ).scale_to_fit_width(2)
        ffhq_2.add(SurroundingRectangle(ffhq_2, color=WHITE, stroke_width=2, buff=0))
        ffhq_2.next_to(axes_list[2], DR, buff=-0.6)
        ffhq_3 = ImageMobject(
            "img/pure_noise.png",
        ).scale_to_fit_width(2)
        ffhq_3.add(SurroundingRectangle(ffhq_3, color=WHITE, stroke_width=2, buff=0))
        ffhq_3.next_to(axes_list[3], UP, buff=0.2)

        particle_0 = dot_groups_smiley[0][0]

        self.play(
            LaggedStart(
                FadeOut(dot_groups_smiley[0]),
                FadeOut(dot_groups_smiley[1]),
                FadeOut(dot_groups_smiley[2]),
                lag_ratio=0.5,
                run_time=2,
            )
        )

        particle_3 = dot_groups_smiley[3][0]
        line_3 = Line(
            particle_3.get_center(),
            ffhq_3.get_corner(DL),
            color=WHITE,
            stroke_width=2,
        )

        self.play(
            LaggedStart(
                Create(line_3),
                FadeIn(ffhq_3),
                lag_ratio=0.8,
            )
        )

        dots_copy = dot_groups_smiley[3].copy()
        self.add(dots_copy)
        self.play(ReplacementTransform(dots_copy, dot_groups_smiley[2]), run_time=2)

        particle_2 = dot_groups_smiley[2][0]
        line_2 = Line(
            particle_2.get_center(),
            ffhq_2.get_corner(UL),
            color=WHITE,
            stroke_width=2,
        )
        self.play(
            LaggedStart(
                Create(line_2),
                FadeIn(ffhq_2),
                lag_ratio=0.8,
            )
        )

        dots_copy = dot_groups_smiley[2].copy()
        self.add(dots_copy)
        self.play(ReplacementTransform(dots_copy, dot_groups_smiley[1]), run_time=2)

        particle_1 = dot_groups_smiley[1][0]
        line_1 = Line(
            particle_1.get_center(),
            ffhq_1.get_corner(UL),
            color=WHITE,
            stroke_width=2,
        )

        self.play(
            LaggedStart(
                Create(line_1),
                FadeIn(ffhq_1),
                lag_ratio=0.8,
            )
        )

        dots_copy = dot_groups_smiley[1].copy()
        self.add(dots_copy)
        self.play(ReplacementTransform(dots_copy, dot_groups_smiley[0]), run_time=2)

        line_0 = Line(
            particle_0.get_center(),
            ffhq_0.get_corner(UL),
            color=WHITE,
            stroke_width=2,
        )

        self.play(LaggedStart(Create(line_0), FadeIn(ffhq_0), lag_ratio=0.8))

        self.play(FadeOut(*self.mobjects, shift=0.5 * DOWN))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene1_5()
    scene.render()
