from manim import *
import numpy as np
import random
import torch

tex_to_color_map = {
    r"f(x, t)": BLUE_D,
    r"g(t)": RED_D,
    r"\Delta t": PURPLE_B,
    r"-\frac{1}{2} \beta(t) x": BLUE_D,
    r"-\frac{1}{2} \beta(t) x_t": BLUE_D,
    r"\sqrt{\beta(t)}": RED_D,
    r"\nabla_x \log p_t(x)": PURPLE_B,
    r"\nabla_x \log p_t(x_t | x_0)": PURPLE_B,
    r"\theta": TEAL_C,
    r"q(x_{t-1} \mid x_t, x_0)": RED_C,
}

tex_to_color_map_2 = {
    r"f(x, t)": BLUE_D,
    r"g(t)": RED_D,
    r"\Delta t": PURPLE_B,
    r"-\frac{1}{2} \beta(t) x_t": BLUE_D,
    r"\sqrt{\beta(t)}": RED_D,
    r"\nabla_x \log p_t(x)": PURPLE_B,
    r"\nabla_x \log p_t(x_t | x_0)": PURPLE_B,
    r"\theta": TEAL_C,
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


class Scene1_10(Scene):
    def construct(self):

        # Comparison score-based / DDPM
        self.next_section(skip_animations=False)

        noising_process = Tex(
            r"Noising Process",
            font_size=36,
            color=WHITE,
        )
        reverse_process = Tex(
            r"Reverse Process",
            font_size=36,
            color=WHITE,
        )
        objective = Tex(
            r"Objective",
            font_size=36,
            color=WHITE,
        )
        output = Tex(
            r"Network Output",
            font_size=36,
            color=WHITE,
        )
        training_loss = Tex(
            r"Training Loss",
            font_size=36,
            color=WHITE,
        )

        categories = (
            VGroup(noising_process, reverse_process, objective, output, training_loss)
            .arrange(DOWN, aligned_edge=LEFT, buff=1.0)
            .to_edge(LEFT, buff=0.25)
        )

        underlines = VGroup(
            Underline(noising_process, color=WHITE),
            Underline(reverse_process, color=WHITE),
            Underline(objective, color=WHITE),
            Underline(output, color=WHITE),
            Underline(training_loss, color=WHITE),
        )

        ddpm_title = Tex("DDPM", font_size=46, color=WHITE)
        ddpm_title.to_edge(UP).shift(LEFT * 3.5)
        ddpm_title_ul = Underline(ddpm_title, color=WHITE)
        score_based_title = Tex("Score-diffusion", font_size=46, color=WHITE)
        score_based_title.to_edge(UP).shift(RIGHT * 3.5)
        score_based_title_ul = Underline(score_based_title, color=WHITE)

        separation_line = DashedLine(
            UP * (config.frame_height / 2),
            DOWN * (config.frame_height / 2),
            buff=0.5,
            color=WHITE,
        )

        table = (
            VGroup(
                VGroup(ddpm_title, ddpm_title_ul),
                separation_line,
                VGroup(score_based_title, score_based_title_ul),
            )
            .arrange(RIGHT, aligned_edge=UP, buff=2.0)
            .to_edge(RIGHT, buff=1.5)
        )

        VGroup(ddpm_title, ddpm_title_ul).to_edge(UP, buff=0.1)
        VGroup(score_based_title, score_based_title_ul).to_edge(UP, buff=0.1)

        self.play(
            FadeIn(ddpm_title, ddpm_title_ul, score_based_title, score_based_title_ul),
            Create(separation_line),
        )

        xt_circle = Circle(radius=0.5, color=WHITE).move_to(RIGHT * 1.5)
        xt1_circle = Circle(radius=0.5, color=WHITE).move_to(LEFT * 1.5)

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
            color=RED,
            angle=-PI / 2,
        )
        q_label = MathTex(
            r"q(x_t \mid x_{t-1})",
            tex_to_color_map=tex_to_color_map,
            color=RED,
            font_size=42,
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
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        ).next_to(p_arrow, DOWN, buff=0.1)

        q_arrow.set_color(RED_C)
        p_arrow.set_color(BLUE_C)

        ddpm_diagram = (
            VGroup(
                xt_circle,
                xt1_circle,
                xt_label,
                xt1_label,
                q_arrow,
                q_label,
                p_arrow,
                p_label,
            )
            .scale_to_fit_width(3)
            .move_to(
                (
                    ddpm_title.get_center()[0],
                    (categories[0].get_center()[1] + categories[1].get_center()[1]) / 2,
                    0,
                )
            )
        )

        # Noising process

        self.play(FadeIn(categories[0], underlines[0]), run_time=1.0)
        self.play(
            LaggedStart(
                FadeIn(xt1_circle, xt1_label),
                Create(q_arrow),
                Write(q_label),
                FadeIn(xt_circle, xt_label),
            ),
            run_time=2,
        )

        arrow_forward = Arrow(
            start=3.5 * LEFT,
            end=3.5 * RIGHT,
        )
        arrow_reverse = Arrow(
            start=3.5 * RIGHT,
            end=3.5 * LEFT,
        ).next_to(arrow_forward, DOWN, buff=1.0)

        ddpm_forward_sde = MathTex(
            r"\mathrm{d}x = -\frac{1}{2} \beta(t) x \mathrm{d}t + \sqrt{\beta(t)} \mathrm{d}W",
            font_size=48,
            tex_to_color_map=tex_to_color_map,
        ).next_to(arrow_forward, UP, buff=0.5)
        ddpm_reverse_sde = MathTex(
            r"\mathrm{d}x = \left[-\frac{1}{2} \beta(t) x - \beta(t) \nabla_x \log p_t(x) \right] \mathrm{d}t + \sqrt{\beta(t)} \mathrm{d}\bar{W}",
            font_size=48,
            tex_to_color_map=tex_to_color_map,
        ).next_to(arrow_reverse, DOWN, buff=0.5)

        score_diagram = (
            VGroup(
                ddpm_forward_sde,
                arrow_forward,
                ddpm_reverse_sde,
                arrow_reverse,
            )
            .scale_to_fit_width(5)
            .move_to(
                (
                    score_based_title.get_center()[0],
                    (categories[0].get_center()[1] + categories[1].get_center()[1]) / 2,
                    0,
                )
            )
        )

        self.play(
            LaggedStart(
                GrowArrow(arrow_forward), Write(ddpm_forward_sde), lag_ratio=0.2
            ),
            run_time=2,
        )

        # Reverse process

        self.play(FadeIn(categories[1], underlines[1]), run_time=1.0)

        self.play(
            LaggedStart(Create(p_arrow), Write(p_label), lag_ratio=0.2), run_time=2
        )

        self.play(
            LaggedStart(
                GrowArrow(arrow_reverse), Write(ddpm_reverse_sde), lag_ratio=0.2
            ),
            run_time=2,
        )

        self.play(FadeIn(categories[2], underlines[2]), run_time=1.0)

        # Objective

        ddpm_objective = MathTex(
            r"\text{Match } p_\theta(x_{t-1} \mid x_t) \text{ to } q(x_{t-1} \mid x_t, x_0)",
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        ).move_to(
            (
                ddpm_title.get_center()[0],
                categories[2].get_center()[1],
                0,
            )
        )

        score_objective = MathTex(
            r"\text{Learn the score } - \nabla_x \log p_t(x)",
            font_size=28,
            tex_to_color_map=tex_to_color_map,
        ).move_to(
            (
                score_based_title.get_center()[0],
                categories[2].get_center()[1],
                0,
            )
        )

        self.play(FadeIn(ddpm_objective), run_time=1)

        self.play(FadeIn(score_objective), run_time=1)

        self.play(FadeIn(categories[3], underlines[3]), run_time=1.0)

        # Network output
        output_ddpm = MathTex(
            r"\epsilon_\theta(x_t, t)",
            font_size=36,
            tex_to_color_map=tex_to_color_map,
        ).move_to((ddpm_title.get_center()[0], categories[3].get_center()[1], 0))
        output_score = MathTex(
            r"s_\theta(x_t, t)",
            font_size=36,
            tex_to_color_map=tex_to_color_map,
        ).move_to(
            (
                score_based_title.get_center()[0],
                categories[3].get_center()[1],
                0,
            )
        )

        self.play(FadeIn(output_ddpm), run_time=1)

        self.play(FadeIn(output_score), run_time=1)

        # Training loss

        self.play(FadeIn(categories[4], underlines[4]), run_time=1.0)

        loss_ddpm = MathTex(
            r"\mathbb{E}_{x_0, t, \epsilon} \left[ \| \epsilon_\theta(x_t, t) - \epsilon \|^2 \right]",
            font_size=32,
            tex_to_color_map=tex_to_color_map,
        ).move_to(
            (
                ddpm_title.get_center()[0],
                categories[4].get_center()[1],
                0,
            )
        )

        loss_score = MathTex(
            r"\mathbb{E}_{x_0, t, \epsilon} \left[ | \sqrt{\beta(t)} s_\theta(x_t, t) + \epsilon \|^2 \right]",
            font_size=32,
            tex_to_color_map=tex_to_color_map,
        ).move_to(
            (
                score_based_title.get_center()[0],
                categories[4].get_center()[1],
                0,
            )
        )

        self.play(FadeIn(loss_ddpm), run_time=1)
        self.play(FadeIn(loss_score), run_time=1)

        # Other valid diffusion SDEs
        self.next_section(skip_animations=False)

        self.play(
            ddpm_forward_sde.animate.move_to(ORIGIN).scale_to_fit_width(8),
            FadeOut(
                categories,
                underlines,
                ddpm_diagram,
                ddpm_reverse_sde,
                arrow_forward,
                arrow_reverse,
                table,
                loss_ddpm,
                loss_score,
                ddpm_objective,
                score_objective,
                output_ddpm,
                output_score,
                score_based_title,
                score_based_title_ul,
                ddpm_title,
                ddpm_title_ul,
            ),
        )

        gen_forward_sde = MathTex(
            r"\mathrm{d}x = f(x, t) \mathrm{d}t + g(t) \mathrm{d}W",
            font_size=46,
            tex_to_color_map=tex_to_color_map,
        )

        self.play(ReplacementTransform(ddpm_forward_sde, gen_forward_sde, run_time=1.5))
        euler = Tex(r"Euler-Maruyama", font_size=36, color=WHITE)
        heun = Tex(
            r"Heun",
            font_size=36,
            color=WHITE,
        )
        runge = Tex(
            r"Runge-Kutta",
            font_size=36,
            color=WHITE,
        )
        dots = Tex(r"\dots", font_size=36, color=WHITE)
        solvers = VGroup(
            euler,
            heun,
            runge,
            dots,
        ).arrange(RIGHT, buff=1.0)

        self.play(
            LaggedStart(
                gen_forward_sde.animate.shift(UP * 1.5),
                Write(solvers),
                lag_ratio=0.2,
                run_time=2.0,
            )
        )

        hyvarinen_score_matching = MathTex(
            r"\mathbb{E}_{x_0, t, \epsilon} \left[ \| s_{\theta} (x_t, t) \|^2 + \mathrm{tr}  \left( \nabla s_{\theta} (x_t, t) \right) \right]",
            font_size=32,
        )

        sliced_score_matching = MathTex(
            r"\mathbb{E}_{x_0, t, \epsilon, v} \left[ 2 v^\top \nabla_x s_\theta(x) v + | v^\top s_\theta(x) |^2 \right]",
            font_size=32,
        )

        objectives = VGroup(
            hyvarinen_score_matching,
            sliced_score_matching,
        ).arrange(RIGHT, buff=0.75)

        hyvarinen_score_matching_label = Tex(
            r"Hyvärinen Score Matching",
            font_size=36,
            color=WHITE,
        ).next_to(hyvarinen_score_matching, UP, buff=0.5)

        sliced_score_matching_label = Tex(
            r"Sliced Score Matching",
            font_size=36,
            color=WHITE,
        ).next_to(sliced_score_matching, UP, buff=0.5)

        VGroup(
            hyvarinen_score_matching_label,
            sliced_score_matching_label,
            hyvarinen_score_matching,
            sliced_score_matching,
        ).shift(DOWN)
        self.play(VGroup(gen_forward_sde, solvers).animate.shift(UP * 2.0))
        self.play(FadeIn(hyvarinen_score_matching_label, hyvarinen_score_matching))
        self.play(FadeIn(sliced_score_matching_label, sliced_score_matching))

        txt = Text("Thanks for watching!").scale(1.2).to_edge(UP, buff=1.0)

        self.play(FadeOut(*self.mobjects, shift=0.5 * DOWN), run_time=1.5)

        self.play(Write(txt, run_time=1.5))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene1_10()
    scene.render()
