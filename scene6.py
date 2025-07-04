from manim import *
from util import read_script
import numpy as np
import random
import torch

tex_to_color_map = {
    r"f(x, t)": BLUE_D,
    r"g(t)": RED_D,
    r"\Delta t": PURPLE_B,
    r"-\frac{1}{2} \beta(t) x": BLUE_D,
    r"\sqrt{\beta(t)}": RED_D,
    r"\nabla_x \log p_t(x)": PURPLE_B,
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


class Scene1_6(MovingCameraScene):
    def construct(self):

        # Present the reverse SDE
        self.next_section(skip_animations=False)

        gen_forward_sde = MathTex(
            r"\mathrm{d}x = f(x, t) \mathrm{d}t + g(t) \mathrm{d}W",
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        )
        gen_reverse_sde = MathTex(
            r"\mathrm{d}x = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x) \right] \mathrm{d}t + g(t) \mathrm{d}\bar{W}",
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        )

        arrow_forward = Arrow(
            3 * LEFT,
            3 * RIGHT,
            buff=0.0,
            color=WHITE,
            stroke_width=2,
        )
        arrow_reverse = Arrow(
            3 * RIGHT,
            3 * LEFT,
            buff=0.0,
            color=WHITE,
            stroke_width=2,
        )

        arrows = (
            VGroup(arrow_forward, arrow_reverse).arrange(DOWN, buff=1.0).move_to(ORIGIN)
        )
        gen_forward_sde.next_to(arrow_forward, UP, buff=0.5)
        gen_reverse_sde.next_to(arrow_reverse, DOWN, buff=0.5)

        gen_forward_sde_label = Tex(
            "Forward SDE",
            font_size=42,
            color=WHITE,
        ).next_to(gen_forward_sde, UP, buff=0.75)
        gen_forward_sde_label_ul = Underline(gen_forward_sde_label)
        gen_reverse_sde_label = Tex(
            "Reverse SDE",
            font_size=42,
            color=WHITE,
        ).next_to(gen_reverse_sde, DOWN, buff=0.75)
        gen_reverse_sde_label_ul = Underline(gen_reverse_sde_label)

        p_x = Circle(
            radius=0.6,
            color=WHITE,
            stroke_width=2,
        )
        p_x.add(
            MathTex(
                r"p(x)",
                font_size=38,
                tex_to_color_map=tex_to_color_map,
            )
        )
        p_x.next_to(arrows, LEFT, buff=0.75)
        prior = Circle(
            radius=0.6,
            color=WHITE,
            stroke_width=2,
        )
        prior.add(
            MathTex(
                r"p_T(x)",
                font_size=38,
                tex_to_color_map=tex_to_color_map,
            )
        )
        prior.next_to(arrows, RIGHT, buff=0.75)

        figure = VGroup(
            gen_forward_sde,
            gen_reverse_sde,
            arrows,
            gen_forward_sde_label,
            gen_reverse_sde_label,
            gen_forward_sde_label_ul,
            gen_reverse_sde_label_ul,
            p_x,
            prior,
        ).to_edge(UP, buff=0.5)

        self.play(FadeIn(p_x))
        self.play(
            LaggedStart(
                Create(arrow_forward),
                Write(gen_forward_sde),
                LaggedStart(
                    FadeIn(gen_forward_sde_label),
                    Create(gen_forward_sde_label_ul),
                    lag_ratio=0.8,
                ),
                lag_ratio=0.2,
            ),
            run_time=1.5,
        )
        self.play(FadeIn(prior))

        self.play(
            LaggedStart(
                Create(arrow_reverse),
                Write(gen_reverse_sde),
                LaggedStart(
                    FadeIn(gen_reverse_sde_label),
                    Create(gen_reverse_sde_label_ul),
                    lag_ratio=0.8,
                ),
                lag_ratio=0.2,
            ),
            run_time=3,
        )

        n_samples = 400
        axes_p = (
            Axes(
                x_range=(-3, 3),
                y_range=(-3, 3),
                axis_config={"color": WHITE},
                tips=False,
            )
            .scale_to_fit_width(2.5)
            .next_to(arrows, LEFT, buff=0.75)
        )
        axes_prior = (
            Axes(
                x_range=(-3, 3),
                y_range=(-3, 3),
                axis_config={"color": WHITE},
                tips=False,
            )
            .scale_to_fit_width(2.5)
            .next_to(arrows, RIGHT, buff=0.75)
        )

        X_smiley, y_smiley = make_smiley(
            n_samples=n_samples, noise=0, random_state=seed
        )
        X_smiley = X_smiley - np.mean(X_smiley, axis=0)

        snapshots_smiley = simulate_sde(X_smiley, beta, save_steps=[0, 1000])
        dot_groups_smiley = VGroup()

        for i, (X_snapshot, ax) in enumerate(
            zip(snapshots_smiley, [axes_p, axes_prior])
        ):
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

        ddpm_forward_sde = MathTex(
            r"\mathrm{d}x = -\frac{1}{2} \beta(t) x \mathrm{d}t + \sqrt{\beta(t)} \mathrm{d}W",
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        ).move_to(gen_forward_sde)

        self.play(ReplacementTransform(prior, dot_groups_smiley[1]), run_time=2)

        self.wait(1)
        self.play(ReplacementTransform(p_x, dot_groups_smiley[0]), run_time=2)

        ddpm_reverse_sde = MathTex(
            r"\mathrm{d}x = \left[-\frac{1}{2} \beta(t) x - \beta(t) \nabla_x \log p_t(x) \right] \mathrm{d}t + \sqrt{\beta(t)} \mathrm{d}\bar{W}",
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        ).move_to(gen_reverse_sde)

        self.wait(0.5)
        self.play(ReplacementTransform(gen_forward_sde, ddpm_forward_sde))
        self.wait(1)
        self.play(ReplacementTransform(gen_reverse_sde, ddpm_reverse_sde))

        # Transition onto the notion of score
        self.next_section(skip_animations=False)

        self.play(Circumscribe(ddpm_reverse_sde[5], color=RED_B, run_time=1.5))

        self.wait(4)
        self.play(Circumscribe(ddpm_reverse_sde[1], color=BLUE_B, run_time=1.5))

        self.wait(1)
        self.play(Circumscribe(ddpm_reverse_sde[3], color=PURPLE_B, run_time=1.5))

        self.play(
            LaggedStart(
                FadeOut(
                    ddpm_forward_sde,
                    gen_forward_sde_label,
                    gen_forward_sde_label_ul,
                    arrows,
                    dot_groups_smiley[0],
                    dot_groups_smiley[1],
                    gen_reverse_sde_label,
                    gen_reverse_sde_label_ul,
                ),
                ddpm_reverse_sde.animate.to_edge(UP, buff=0.5),
            )
        )

        rect_score = SurroundingRectangle(ddpm_reverse_sde[3], buff=0.1, color=PURPLE_B)
        score_label = Tex(
            r"Score Functions",
            font_size=42,
            color=PURPLE_B,
        ).next_to(rect_score, DOWN, buff=0.5)

        self.play(Create(rect_score))
        self.play(
            Write(score_label),
            run_time=1.5,
        )

        self.play(FadeOut(rect_score, score_label))
        self.wait(1)


if __name__ == "__main__":
    scene = Scene1_6()
    scene.render()
