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


def generate_gaussian_mixture_samples(n_samples, centers, covariances, weights):
    """
    Generate samples from a 2D Gaussian mixture.
    """
    samples = []
    n_samples_per_mode = np.random.multinomial(n_samples, weights)

    for i, (center, cov, n_mode_samples) in enumerate(
        zip(centers, covariances, n_samples_per_mode)
    ):
        mode_samples = np.random.multivariate_normal(center, cov, n_mode_samples)
        samples.append(mode_samples)

    all_samples = np.vstack(samples)

    return all_samples


def gaussian_mixture_score(x, centers, covariances, weights):
    """
    Compute the score function (gradient of log-density) for a 2D Gaussian mixture.
    """
    centers = np.array(centers)
    covariances = np.array(covariances)
    weights = np.array(weights)

    # Compute densities for each component
    densities = []
    for i in range(len(centers)):
        diff = x - centers[i]
        inv_cov = np.linalg.inv(covariances[i])
        det_cov = np.linalg.det(covariances[i])
        density = (
            weights[i]
            * np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=1))
            / (2 * np.pi * np.sqrt(det_cov))
        )
        densities.append(density)

    densities = np.array(densities).T  # Shape: (n_points, n_components)
    total_density = np.sum(densities, axis=1, keepdims=True)

    # Compute weighted score
    score = np.zeros_like(x)
    for i in range(len(centers)):
        diff = x - centers[i]
        inv_cov = np.linalg.inv(covariances[i])
        component_score = -diff @ inv_cov
        weight = densities[:, i : i + 1] / total_density
        score += weight * component_score

    return score


def create_mixture_contours(axes, centers, covariances, n_ellipses=4):
    """
    Draw covariance ellipses whose size *and* tilt are correct in screen
    coordinates.
    """
    ellipses = VGroup()

    # how many pixels correspond to one data unit
    x_unit = axes.x_axis.get_unit_size()  # or axes.x_axis.unit_size with older Manim
    y_unit = axes.y_axis.get_unit_size()

    for center, cov in zip(centers, covariances):
        # eigendecomposition (use eigh because cov is symmetric)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # --- orientation -----------------------------------------------------
        # principal axis in data space
        v = vecs[:, 0]
        # same vector in *screen* space
        v_screen = np.array([v[0] * x_unit, v[1] * y_unit])
        angle = np.arctan2(v_screen[1], v_screen[0])

        alpha = 0.02
        # --- draw several iso-σ ellipses -------------------------------------
        for k in np.linspace(1, 3, n_ellipses):
            width = 2 * np.sqrt(vals[0]) * k * x_unit  # major axis
            height = 2 * np.sqrt(vals[1]) * k * y_unit  # minor axis

            e = (
                Ellipse(
                    width=width,
                    height=height,
                    color=PURPLE_C,
                    fill_opacity=alpha,
                    stroke_width=0,
                )
                .rotate(angle)  # now the angle is right
                .move_to(axes.c2p(*center))
            )
            ellipses.add(e)

    return ellipses


def create_smoothed_mixture(centers, covariances, weights, t, beta_func):
    """Create a smoothed version of the Gaussian mixture using DDPM forward process."""
    beta_t = beta_func(t)
    alpha_t = 1 - beta_t

    # Apply forward process to centers
    smoothed_centers = []
    for center in centers:
        noise = np.random.normal(0, 1, size=2)
        smoothed_center = np.sqrt(alpha_t) * np.array(center)
        smoothed_centers.append(smoothed_center.tolist())

    # Increase covariances due to added noise
    smoothed_covariances = []
    for cov in covariances:
        noise_var = beta_t * np.eye(2)
        smoothed_cov = alpha_t * cov + noise_var
        smoothed_covariances.append(smoothed_cov)

    return smoothed_centers, smoothed_covariances, weights.copy()


class Scene1_7(MovingCameraScene):
    def construct(self):
        # Read script
        script = read_script("scripts/scene7.txt")

        # Score of mixture of Gaussians
        self.next_section(skip_animations=False)

        axes = Axes(
            x_range=(-5, 5, 1),
            y_range=(-5, 5, 1),
            axis_config={"color": WHITE},
        )

        centers = [(-2, 2), (2, -2)]
        covariances = [
            np.array([[0.5, -0.1], [-0.1, 0.45]]),
            np.array([[0.4, 0.1], [0.1, 0.5]]),
        ]
        weights = [0.5, 0.5]

        coords = generate_gaussian_mixture_samples(
            n_samples=2000, centers=centers, covariances=covariances, weights=weights
        )
        points = VGroup()

        for p in coords:
            dot = Dot(point=axes.c2p(*p), color=BLUE_D, radius=0.025, fill_opacity=1.0)
            points.add(dot)

        p_x = MathTex(r"p(x)", color=PURPLE_B, font_size=42, z_index=3).to_edge(
            UP,
            buff=0.5,
        )
        p_x_rect = SurroundingRectangle(
            p_x, color=WHITE, fill_color=BLACK, fill_opacity=0.8
        )
        p_x_rect.set_z_index(-1)

        self.play(FadeIn(points), FadeIn(p_x, p_x_rect))

        level_lines = create_mixture_contours(
            axes, centers, covariances, n_ellipses=100
        )
        level_lines.set_z_index(-2)

        self.play(
            Create(level_lines),
        )

        def score_func(pos):
            data_pos = np.array([axes.p2c(pos)])
            score = gaussian_mixture_score(data_pos, centers, covariances, weights)
            score_scaled = score[0] * 0.3
            return axes.c2p(*(data_pos[0] + score_scaled)) - pos

        vector_field = ArrowVectorField(
            score_func,
            x_range=[-7, 7, 0.5],
            y_range=[-5, 5, 0.5],
            colors=[
                BLUE_D,
                YELLOW,
                RED,
            ],
            min_color_scheme_value=0.0,
            max_color_scheme_value=3.0,
            length_func=lambda x: 0.35 * sigmoid(x),
            vector_config={"max_tip_length_to_length_ratio": 0.3},
        ).set_z_index(-2)

        self.play(FadeOut(points), run_time=1)
        self.play(Create(vector_field), run_time=2)

        self.play(Uncreate(vector_field))

        # Smoothing distributions
        self.next_section(skip_animations=False)

        def create_mixture_visualization(
            axes, centers, covariances, weights, color=BLUE_D
        ):
            """Create contours and vector field for a Gaussian mixture."""
            # Create contours
            level_lines = create_mixture_contours(
                axes, centers, covariances, n_ellipses=100
            )
            level_lines.set_z_index(-1)

            # Create vector field
            def score_func(pos):
                data_pos = np.array([axes.p2c(pos)])
                score = gaussian_mixture_score(data_pos, centers, covariances, weights)
                score_scaled = score[0] * 0.3
                return axes.c2p(*(data_pos[0] + score_scaled)) - pos

            vector_field = ArrowVectorField(
                score_func,
                x_range=[-7, 7, 0.5],
                y_range=[-5, 5, 0.5],
                colors=[color, YELLOW, RED],
                min_color_scheme_value=0.0,
                max_color_scheme_value=3.0,
                length_func=lambda x: 0.35 * sigmoid(x),
                vector_config={"max_tip_length_to_length_ratio": 0.3},
            )

            return level_lines, vector_field

        # Create smoothed versions at different time steps
        t1, t2 = 800, 1800
        smoothed_centers_1, smoothed_covariances_1, smoothed_weights_1 = (
            create_smoothed_mixture(centers, covariances, weights, t1, beta)
        )
        smoothed_centers_2, smoothed_covariances_2, smoothed_weights_2 = (
            create_smoothed_mixture(centers, covariances, weights, t2, beta)
        )

        # Create visualizations for each smoothing level
        smoothed_level_lines_1, smoothed_vector_field_1 = create_mixture_visualization(
            axes,
            smoothed_centers_1,
            smoothed_covariances_1,
            smoothed_weights_1,
            GREEN_D,
        )
        smoothed_level_lines_2, smoothed_vector_field_2 = create_mixture_visualization(
            axes,
            smoothed_centers_2,
            smoothed_covariances_2,
            smoothed_weights_2,
            PURPLE_D,
        )

        ddpm_forward_sde = MathTex(
            r"\mathrm{d}x = -\frac{1}{2} \beta(t) x \mathrm{d}t + \sqrt{\beta(t)} \mathrm{d}W",
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        ).to_edge(UP, buff=0.5)
        ddpm_forward_sde_rect = SurroundingRectangle(
            ddpm_forward_sde, color=WHITE, fill_color=BLACK, fill_opacity=0.8
        )
        ddpm_forward_sde_rect.set_z_index(-1)

        self.play(
            ReplacementTransform(p_x, ddpm_forward_sde),
            ReplacementTransform(p_x_rect, ddpm_forward_sde_rect),
            run_time=2,
        )

        p_x_t = MathTex(r"p_t (x)", color=PURPLE_B, font_size=42, z_index=3).to_edge(
            UP, buff=0.5
        )
        p_x_t_rect = SurroundingRectangle(
            p_x_t, color=WHITE, fill_color=BLACK, fill_opacity=0.8
        )
        p_x_t_rect.set_z_index(-1)

        self.play(
            Transform(ddpm_forward_sde, p_x_t),
            Transform(ddpm_forward_sde_rect, p_x_t_rect),
            run_time=2,
        )

        p_x_1 = MathTex(
            r"p_{100} (x)", font_size=42, color=PURPLE_B, z_index=3
        ).to_edge(UP, buff=0.5)
        p_x_1_rect = SurroundingRectangle(
            p_x_1, color=WHITE, fill_color=BLACK, fill_opacity=0.8
        )
        p_x_1_rect.set_z_index(-1)

        p_x_2 = MathTex(
            r"p_{500} (x)", font_size=42, color=PURPLE_B, z_index=3
        ).to_edge(UP, buff=0.5)

        self.play(FadeOut(ddpm_forward_sde, ddpm_forward_sde_rect))

        level_lines_cp = level_lines.copy()
        self.play(
            LaggedStart(FadeIn(p_x_1_rect), Write(p_x_1), lag_ratio=0.2),
            Transform(level_lines, smoothed_level_lines_1),
            run_time=3,
        )

        self.play(
            ReplacementTransform(p_x_1, p_x_2),
            Transform(level_lines, smoothed_level_lines_2),
            run_time=3,
        )

        self.play(
            FadeOut(p_x_2, p_x_1_rect),
            Transform(level_lines, level_lines_cp),
        )

        vector_field = ArrowVectorField(
            score_func,
            x_range=[-7, 7, 0.5],
            y_range=[-5, 5, 0.5],
            colors=[
                BLUE_D,
                YELLOW,
                RED,
            ],
            min_color_scheme_value=0.0,
            max_color_scheme_value=3.0,
            length_func=lambda x: 0.35 * sigmoid(x),
            vector_config={"max_tip_length_to_length_ratio": 0.3},
        ).set_z_index(-2)

        score_function = MathTex(
            r"\nabla_x \log p (x)", font_size=42, color=PURPLE_B, z_index=3
        ).to_edge(UP, buff=0.5)
        score_function_rect = SurroundingRectangle(
            score_function, color=WHITE, fill_color=BLACK, fill_opacity=0.8
        )
        score_function_rect.set_z_index(-1)

        self.play(
            FadeIn(score_function_rect),
            Write(score_function),
            run_time=1.0,
        )
        self.play(
            Create(vector_field),
            run_time=3,
        )

        score_function_1 = MathTex(
            r"\nabla_x \log p_{100} (x)", font_size=42, color=PURPLE_B, z_index=3
        ).to_edge(UP, buff=0.5)
        score_function_1_rect = SurroundingRectangle(
            score_function_1, color=WHITE, fill_color=BLACK, fill_opacity=0.8
        )
        score_function_1_rect.set_z_index(-1)

        score_function_2 = MathTex(
            r"\nabla_x \log p_{500} (x)", font_size=42, color=PURPLE_B, z_index=3
        ).to_edge(UP, buff=0.5)
        score_function_2_rect = SurroundingRectangle(
            score_function_2, color=WHITE, fill_color=BLACK, fill_opacity=0.8
        )
        score_function_2_rect.set_z_index(1)

        self.play(
            Transform(score_function, score_function_1),
            Transform(score_function_rect, score_function_1_rect),
            Transform(vector_field, smoothed_vector_field_1),
            Transform(level_lines, smoothed_level_lines_1),
            run_time=3,
        )
        self.play(
            Transform(score_function, score_function_2),
            Transform(score_function_rect, score_function_2_rect),
            Transform(vector_field, smoothed_vector_field_2),
            Transform(level_lines, smoothed_level_lines_2),
            run_time=3,
        )

        # Transition to learning the score
        self.next_section(skip_animations=False)

        score_function_final = MathTex(
            r"\nabla_x \log p_t (x)",
            r"= ?",
            font_size=42,
            color=PURPLE_B,
            z_index=3,
        ).move_to(ORIGIN)

        self.play(
            LaggedStart(
                AnimationGroup(
                    Uncreate(level_lines),
                    FadeOut(vector_field, score_function_rect),
                ),
                Transform(score_function, score_function_final),
                lag_ratio=0.2,
            ),
            run_time=2,
        )

        question = Tex(
            r"How do we learn this score ?", font_size=42, color=WHITE
        ).to_edge(UP, buff=1.5)

        self.play(Write(question), run_time=1)

        self.play(FadeOut(question, score_function, shift=0.5 * DOWN))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene1_7()
    scene.render()
