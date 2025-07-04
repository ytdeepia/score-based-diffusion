from manim import *
import numpy as np
import random
import torch

tex_to_color_map = {r"f(x, t)": BLUE_D, r"g(t)": RED_D, r"\Delta t": PURPLE_B}

# Set seeds for reproducibility
seed = 2
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class Scene1_4(MovingCameraScene):
    def construct(self):
        # From DDPM markov chain to Ornstein-Uhlenbeck process
        self.next_section(skip_animations=False)

        markov_formula = MathTex(
            r"x_i = \sqrt{1 - \beta_i} x_{i-1} + \sqrt{\beta_i} \epsilon_{i-1}",
            color=WHITE,
            font_size=36,
        )
        markov_rect = SurroundingRectangle(markov_formula, color=WHITE, buff=0.2)
        markov_label = Tex("DDPM Markov Chain", font_size=36, color=WHITE).next_to(
            markov_rect, UP, buff=0.5
        )
        markov_vg = VGroup(markov_formula, markov_rect, markov_label)
        self.play(Write(markov_formula))
        self.play(Create(markov_rect))
        self.play(Write(markov_label))
        diffusion_sde = MathTex(
            r"\mathrm{d}x = f(x, t) \mathrm{d}t + g(t) \mathrm{d}W",
            tex_to_color_map=tex_to_color_map,
            font_size=36,
        )
        diffusion_sde_rect = SurroundingRectangle(diffusion_sde, color=WHITE, buff=0.2)
        diffusion_sde_label = Tex("Itô SDE", font_size=36, color=WHITE).next_to(
            diffusion_sde_rect, DOWN, buff=0.5
        )

        sde_vg = VGroup(diffusion_sde, diffusion_sde_rect, diffusion_sde_label)

        self.play(markov_vg.animate.to_edge(UP, buff=0.5))

        sde_vg.next_to(markov_vg, DOWN, buff=2.0)
        arrow = Arrow(
            start=markov_rect.get_bottom(),
            end=diffusion_sde_rect.get_top(),
            color=WHITE,
            buff=0.2,
        )

        self.play(
            LaggedStart(
                Create(arrow),
                LaggedStart(Write(diffusion_sde), Create(diffusion_sde_rect)),
            )
        )
        self.play(Write(diffusion_sde_label))

        self.play(FadeOut(markov_rect, arrow, sde_vg))

        beta_tilde = MathTex(
            r"\tilde{\beta}_i = N \beta_i",
            font_size=36,
        )

        markov_beta_tilde = MathTex(
            r"x_i = \sqrt{1 - \frac{\tilde{\beta}_i}{N}} x_{i-1} + \sqrt{\frac{\tilde{\beta}_i}{N}} \epsilon_{i-1}",
            font_size=36,
        )
        VGroup(beta_tilde, markov_beta_tilde).arrange(RIGHT, buff=1.0).next_to(
            markov_formula, DOWN, buff=1.0
        )

        self.play(
            Write(beta_tilde),
        )

        self.play(
            Write(markov_beta_tilde),
        )

        delta_t = MathTex(
            r"\Delta t = \frac{1}{N}",
            tex_to_color_map=tex_to_color_map,
            font_size=36,
        )
        x_eq = MathTex(
            r"x_{i} = x(t + \Delta t)",
            tex_to_color_map=tex_to_color_map,
            font_size=36,
        )
        beta_eq = MathTex(
            r"\tilde{\beta}_i = \beta(t + \Delta t)",
            tex_to_color_map=tex_to_color_map,
            font_size=36,
        )

        VGroup(delta_t, x_eq, beta_eq).arrange(RIGHT, buff=1.0).next_to(
            VGroup(beta_tilde, markov_beta_tilde), DOWN, buff=1.0
        )
        self.play(
            Write(delta_t),
        )

        self.play(
            Write(x_eq),
        )
        self.play(
            Write(beta_eq),
        )

        markov_increment = MathTex(
            r"x(t + \Delta t) =",
            r"\sqrt{1 - \beta (t + \Delta t) \Delta t}",
            r"x(t) + \sqrt{\beta (t + \Delta t) \Delta t} \epsilon(t)",
            tex_to_color_map=tex_to_color_map,
            font_size=36,
        ).move_to(VGroup(beta_tilde, markov_beta_tilde))

        self.play(FadeOut(beta_tilde, markov_beta_tilde))
        self.play(Write(markov_increment))

        markov_dl = MathTex(
            r"x(t + \Delta t) \approx x(t) - \frac{1}{2} \beta(t + \Delta t) x(t) \Delta t + \sqrt{\beta(t + \Delta t) \Delta t} \epsilon(t)",
            font_size=36,
            tex_to_color_map=tex_to_color_map,
        ).next_to(markov_increment, DOWN, buff=0.5)

        self.wait(3)
        rect = SurroundingRectangle(markov_increment[3:5], color=WHITE, buff=0.1)
        self.play(Create(rect))

        self.wait(2)
        taylor_expansion = Tex(
            "First-order Taylor expansion", font_size=36, color=WHITE
        ).next_to(rect, DOWN, buff=0.5)

        self.play(
            LaggedStart(
                FadeOut(delta_t, x_eq, beta_eq),
                Write(taylor_expansion),
                lag_ratio=0.8,
            ),
            run_time=2,
        )

        self.play(
            LaggedStart(
                FadeOut(taylor_expansion, rect), Write(markov_dl), lag_ratio=0.8
            ),
            run_time=2,
        )

        markov_dl_2 = MathTex(
            r"x(t + \Delta t) \approx x(t) - \frac{1}{2} \beta(t) x(t) \Delta t + \sqrt{\beta(t) \Delta t} \epsilon(t)",
            font_size=36,
            tex_to_color_map=tex_to_color_map,
        ).next_to(markov_dl, DOWN, buff=0.5)

        self.play(Write(markov_dl_2))

        markov_dl_3 = MathTex(
            r"x(t + \Delta t) - x(t) \approx - \frac{1}{2} \beta(t) x(t) \Delta t + \sqrt{\beta(t) \Delta t} \epsilon(t)",
            font_size=36,
            tex_to_color_map=tex_to_color_map,
        ).next_to(markov_formula, DOWN, buff=1.0)

        self.play(
            LaggedStart(
                FadeOut(markov_increment, markov_dl, markov_dl_2),
                Write(markov_dl_3),
                lag_ratio=0.8,
            )
        )

        markov_limit = MathTex(
            r"\mathrm{d}x \approx - \frac{1}{2} \beta(t) x(t) \mathrm{d}t + \sqrt{\beta(t)}",
            r"\sqrt{\mathrm{d}t} \epsilon(t)",
            font_size=36,
        ).next_to(markov_dl_3, DOWN, buff=1.5)

        arrow = Arrow(
            start=markov_dl_3.get_bottom(),
            end=markov_limit.get_top(),
            color=WHITE,
            buff=0.2,
        )
        limit = MathTex(
            r"\lim_{\Delta t \to 0}",
            font_size=36,
            tex_to_color_map=tex_to_color_map,
        ).next_to(arrow, RIGHT, buff=0.5)

        self.play(
            LaggedStart(
                LaggedStart(GrowArrow(arrow), Write(limit), lag_ratio=0.2),
                Write(markov_limit),
                lag_ratio=0.8,
            ),
            run_time=2,
        )

        self.wait(1.5)
        self.play(Circumscribe(markov_limit[1], color=RED_D))
        self.wait(4)
        browian_eq = MathTex(
            r"\mathrm{d}W = \sqrt{\mathrm{d}t} \epsilon(t)",
            font_size=36,
            tex_to_color_map=tex_to_color_map,
        ).next_to(arrow, LEFT, buff=0.5)

        self.play(Write(browian_eq))

        markov_brownian = MathTex(
            r"\mathrm{d}x =",
            r"- \frac{1}{2} \beta(t) x",
            r"\mathrm{d}t + ",
            r"\sqrt{\beta(t)}",
            r"\mathrm{d}W",
            font_size=36,
            tex_to_color_map=tex_to_color_map,
        ).move_to(markov_limit)

        self.play(Transform(markov_limit, markov_brownian))

        self.play(
            LaggedStart(
                FadeOut(arrow, limit, browian_eq, markov_dl_3),
                markov_limit.animate.next_to(markov_formula, DOWN, buff=1.5),
            )
        )

        drift_label = Tex("Drift", font_size=36, color=BLUE_D).next_to(
            markov_limit[1], UP, buff=0.5
        )
        drift_expression = MathTex(
            r"f(x, t)",
            font_size=36,
            tex_to_color_map=tex_to_color_map,
        ).next_to(markov_limit[1], DOWN, buff=0.5)

        diffusion_label = (
            Tex("Diffusion", font_size=36, color=RED_D)
            .next_to(drift_label, RIGHT, aligned_edge=DOWN)
            .set_x(markov_limit[3].get_x())
        )
        diffusion_expression = (
            MathTex(
                r"g(t)",
                font_size=36,
                tex_to_color_map=tex_to_color_map,
            )
            .next_to(drift_expression, RIGHT, aligned_edge=DOWN)
            .set_x(markov_limit[3].get_x())
        )

        self.play(
            markov_limit[1].animate.set_color(BLUE_D),
        )
        self.play(Write(drift_label))
        self.play(Write(drift_expression))
        self.play(markov_limit[3].animate.set_color(RED_D))
        self.play(Write(diffusion_label))
        self.play(Write(diffusion_expression))

        self.play(
            LaggedStart(
                FadeOut(
                    drift_label,
                    drift_expression,
                    diffusion_label,
                    diffusion_expression,
                    markov_formula,
                    markov_label,
                ),
                markov_limit.animate.to_edge(UP, buff=0.5),
                lag_ratio=0.8,
            ),
            run_time=2,
        )

        self.wait(1)


if __name__ == "__main__":
    scene = Scene1_4()
    scene.render()
