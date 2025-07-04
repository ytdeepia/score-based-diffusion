from manim import *
import numpy as np
import random
import torch

tex_to_color_map = {
    r"f(x, t)": BLUE_D,
    r"g(t)": RED_D,
}

# Set seeds for reproducibility
seed = 2
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def mixture_gaussian(x, mus, sigmas, weights):
    return sum(w * gaussian(x, mu, sigma) for mu, sigma, w in zip(mus, sigmas, weights))


class Scene1_2(Scene):
    def construct(self):
        # Show the last video with the mascot on top
        self.next_section(skip_animations=False)

        why_text = Tex("Why another video on diffusion models?", font_size=42).to_edge(
            UP
        )
        why_text_ul = Underline(why_text, color=WHITE, buff=0.1)

        self.play(
            LaggedStart(Create(why_text_ul), Write(why_text), lag_ratio=0.1),
            run_time=2,
        )

        # DDPM formulation
        self.next_section(skip_animations=False)

        line_ddpm = NumberLine(
            x_range=[2011, 2025],
            length=config.frame_width * 0.8,
            color=WHITE,
            include_numbers=False,
            include_ticks=False,
            include_tip=True,
        )

        vae_text = Tex("VAE", font_size=26, color=WHITE, z_index=2).move_to(
            line_ddpm.n2p(2013),
        )
        diffusion_text = Tex("Diffusion", font_size=26, color=WHITE, z_index=2).move_to(
            line_ddpm.n2p(2015),
        )
        ddpm_text = Tex("DDPM", font_size=26, color=WHITE, z_index=2).move_to(
            line_ddpm.n2p(2020),
        )
        vae_rect = SurroundingRectangle(
            vae_text,
            color=WHITE,
            fill_color=BLACK,
            fill_opacity=1,
            buff=0.25,
            corner_radius=0.1,
        )
        diffusion_rect = SurroundingRectangle(
            diffusion_text,
            color=WHITE,
            fill_color=BLACK,
            fill_opacity=1,
            buff=0.25,
            corner_radius=0.1,
        )
        ddpm_rect = SurroundingRectangle(
            ddpm_text,
            color=WHITE,
            fill_color=BLACK,
            fill_opacity=1,
            buff=0.25,
            corner_radius=0.1,
        )
        vae_date = Tex("2013", font_size=24, color=WHITE).next_to(vae_rect, DOWN)
        diffusion_date = Tex("2015", font_size=24, color=WHITE).next_to(
            diffusion_rect, DOWN
        )

        ddpm_date = Tex("2020", font_size=24, color=WHITE).next_to(ddpm_rect, DOWN)
        vae_author = Tex("D. Kingma", font_size=24, color=WHITE).next_to(vae_rect, UP)
        diffusion_author = Tex("S. Sohl-Dickstein", font_size=24, color=WHITE).next_to(
            diffusion_rect, UP
        )
        ddpm_author = Tex("J. Ho", font_size=24, color=WHITE).next_to(ddpm_rect, UP)

        self.play(FadeOut(why_text_ul, why_text), run_time=1)
        self.play(Create(line_ddpm))

        self.play(
            LaggedStart(
                FadeIn(ddpm_rect),
                FadeIn(ddpm_text, ddpm_date, ddpm_author),
                lag_ratio=0.05,
            )
        )

        self.play(
            LaggedStart(
                FadeIn(diffusion_rect),
                FadeIn(diffusion_text, diffusion_date, diffusion_author),
                lag_ratio=0.05,
            )
        )

        self.play(
            LaggedStart(
                FadeIn(vae_rect),
                FadeIn(vae_text, vae_date, vae_author),
                lag_ratio=0.05,
            )
        )

        ddpm_vg = VGroup(
            line_ddpm,
            vae_rect,
            vae_text,
            vae_date,
            vae_author,
            diffusion_rect,
            diffusion_text,
            diffusion_date,
            diffusion_author,
            ddpm_rect,
            ddpm_text,
            ddpm_date,
            ddpm_author,
        )

        self.play(ddpm_vg.animate.to_edge(UP, buff=0.25), run_time=1)

        tex_to_color_map = {
            r"\theta": BLUE_C,
        }

        xt_circle = Circle(radius=0.5, color=WHITE).move_to(RIGHT * 2.5)
        xt1_circle = Circle(radius=0.5, color=WHITE).move_to(LEFT * 2.5)

        xt_label = MathTex(
            r"x_t",
            color=WHITE,
            font_size=32,
        ).move_to(xt_circle)

        xt1_label = MathTex(
            r"x_{t-1}",
            color=WHITE,
            font_size=32,
        ).move_to(xt1_circle)

        q_arrow = CurvedArrow(
            start_point=xt1_circle.get_top() + UP * 0.05,
            end_point=xt_circle.get_top() + UP * 0.05,
            color=WHITE,
            angle=-PI / 2,
        )
        q_label = MathTex(
            r"q(x_t \mid x_{t-1})",
            tex_to_color_map=tex_to_color_map,
            color=WHITE,
            font_size=32,
        ).next_to(q_arrow, UP, buff=0.1)
        p_arrow = DashedVMobject(
            CurvedArrow(
                start_point=xt_circle.get_bottom() + DOWN * 0.05,
                end_point=xt1_circle.get_bottom() + DOWN * 0.05,
                color=BLUE,
                angle=-PI / 2,
            )
        )
        p_label = MathTex(
            r"p_{\theta}(x_{t-1} \mid x_t)",
            color=WHITE,
            font_size=32,
            tex_to_color_map=tex_to_color_map,
        ).next_to(p_arrow, DOWN, buff=0.1)

        ax2 = Axes(
            x_range=[-4, 4],
            y_range=[0, 1],
            axis_config={"include_tip": False, "stroke_width": 1},
            z_index=2,
        ).scale_to_fit_width(3.5)

        p2_plot = ax2.plot(
            lambda x: gaussian(x, 1.2, 0.75),
            color=BLUE_C,
            fill_opacity=0.8,
            stroke_width=5,
            stroke_color=BLUE_B,
            z_index=1,
        )
        q2_plot = ax2.plot(
            lambda x: gaussian(x, -0.7, 0.45),
            color=RED_C,
            fill_opacity=0.8,
            stroke_width=5,
            stroke_color=RED_B,
            z_index=2,
        )
        q_arrow.set_color(RED_C)
        p_arrow.set_color(BLUE_C)

        ddpm_diagram = VGroup(
            xt_circle,
            xt1_circle,
            xt_label,
            xt1_label,
            q_arrow,
            q_label,
            p_arrow,
            p_label,
            p2_plot,
            q2_plot,
        ).move_to(DOWN)

        self.play(
            LaggedStart(
                FadeIn(xt1_circle, xt1_label),
                AnimationGroup(Create(q_arrow), FadeIn(q_label)),
                FadeIn(xt_circle, xt_label),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.wait(1)
        self.play(
            LaggedStart(
                AnimationGroup(Create(p_arrow), FadeIn(p_label)),
                FadeIn(p2_plot, q2_plot),
                lag_ratio=0.4,
            ),
            run_time=2,
        )

        self.play(FadeOut(ddpm_diagram, run_time=1))

        # Score-based diffusion models
        self.next_section(skip_animations=False)

        line_score = NumberLine(
            x_range=[2011, 2025],
            length=config.frame_width * 0.8,
            color=WHITE,
            include_numbers=False,
            include_ticks=False,
            include_tip=True,
        ).shift(DOWN)
        score_matching_text = Tex(
            r"Denoising \\Score \\Matching",
            font_size=26,
            color=WHITE,
            z_index=2,
        ).move_to(line_score.n2p(2013))
        score_matching_rect = SurroundingRectangle(
            score_matching_text,
            color=WHITE,
            fill_color=BLACK,
            fill_opacity=1,
            buff=0.25,
            corner_radius=0.1,
        )
        score_matching_date = Tex("2013", font_size=24, color=WHITE).next_to(
            score_matching_rect, DOWN
        )
        score_matching_author = Tex("P. Vincent", font_size=24, color=WHITE).next_to(
            score_matching_rect, UP
        )
        annealed_langevin = Tex(
            r"Annealed \\Langevin \\Dynamics",
            font_size=26,
            color=WHITE,
            z_index=2,
        ).move_to(line_score.n2p(2019))
        annealed_langevin_rect = SurroundingRectangle(
            annealed_langevin,
            color=WHITE,
            fill_color=BLACK,
            fill_opacity=1,
            buff=0.25,
            corner_radius=0.1,
        )
        annealed_langevin_date = Tex("2019", font_size=24, color=WHITE).next_to(
            annealed_langevin_rect, DOWN
        )
        annealed_langevin_author = Tex("Y. Song", font_size=24, color=WHITE).next_to(
            annealed_langevin_rect, UP
        )

        score_diffusion = Tex(
            r"Score-based \\Diffusion \\Models",
            font_size=26,
            color=WHITE,
            z_index=2,
        ).move_to(line_score.n2p(2021))
        score_diffusion_rect = SurroundingRectangle(
            score_diffusion,
            color=WHITE,
            fill_color=BLACK,
            fill_opacity=1,
            buff=0.25,
            corner_radius=0.1,
        )
        score_diffusion_date = Tex("2021", font_size=24, color=WHITE).next_to(
            score_diffusion_rect, DOWN
        )

        score_diffusion_rect = SurroundingRectangle(
            VGroup(annealed_langevin, score_diffusion),
            color=WHITE,
            fill_color=BLACK,
            fill_opacity=1,
            buff=0.25,
            corner_radius=0.1,
        )

        self.play(Create(line_score))

        self.play(
            LaggedStart(
                FadeIn(annealed_langevin_rect),
                FadeIn(
                    annealed_langevin,
                    annealed_langevin_date,
                    annealed_langevin_author,
                ),
                lag_ratio=0.05,
            )
        )
        self.wait(4)
        self.play(
            LaggedStart(
                FadeIn(score_matching_rect),
                FadeIn(score_matching_text, score_matching_date, score_matching_author),
                lag_ratio=0.05,
            )
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    Transform(
                        annealed_langevin_rect,
                        score_diffusion_rect,
                    ),
                    annealed_langevin_author.animate.next_to(score_diffusion_rect, UP),
                ),
                FadeIn(score_diffusion, score_diffusion_date),
                lag_ratio=0.05,
            ),
            run_time=2,
        )

        score_vg = VGroup(
            line_score,
            score_matching_rect,
            score_matching_text,
            score_matching_date,
            score_matching_author,
            annealed_langevin_rect,
            annealed_langevin,
            annealed_langevin_date,
            annealed_langevin_author,
            score_diffusion,
            score_diffusion_date,
        )
        txt_same = Tex(
            "They both describe the same objects!", font_size=46, color=WHITE
        ).shift(0.5 * UP)
        same_ul = Underline(txt_same, color=WHITE, buff=0.1)
        self.play(
            LaggedStart(
                score_vg.animate.shift(DOWN),
                LaggedStart(Create(same_ul), Write(txt_same), lag_ratio=0.1),
                lag_ratio=0.3,
            ),
            run_time=3,
        )

        self.play(FadeOut(*self.mobjects, shift=0.5 * DOWN))
        self.wait(1)


if __name__ == "__main__":
    scene = Scene1_2()
    scene.render()
