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


class Scene1_8(Scene):
    def construct(self):

        # The neural network
        self.next_section(skip_animations=False)

        title = Tex("The network architecture", font_size=42, color=WHITE).to_edge(
            UP, buff=1.0
        )
        title_ul = Underline(title, color=WHITE, buff=0.1)

        encod_1 = RoundedRectangle(
            width=1,
            height=3,
            color=BLUE_C,
            fill_color=BLUE_C,
            fill_opacity=0.8,
            corner_radius=0.05,
        )

        encod_2 = RoundedRectangle(
            width=0.5,
            height=2.5,
            color=BLUE_C,
            fill_color=BLUE_C,
            fill_opacity=0.8,
            corner_radius=0.05,
        ).next_to(encod_1, RIGHT, buff=0.5, aligned_edge=DOWN)

        encod_3 = RoundedRectangle(
            width=0.5,
            height=2,
            color=BLUE_C,
            fill_color=BLUE_C,
            fill_opacity=0.8,
            corner_radius=0.05,
        ).next_to(encod_2, RIGHT, buff=0.5, aligned_edge=DOWN)

        bottleneck = RoundedRectangle(
            width=3,
            height=1,
            color=RED_D,
            fill_color=RED_D,
            fill_opacity=0.8,
            corner_radius=0.05,
        ).next_to(encod_3, RIGHT, buff=0.5, aligned_edge=DOWN)
        decod_3 = RoundedRectangle(
            width=0.5,
            height=2,
            color=BLUE_C,
            fill_color=BLUE_C,
            fill_opacity=0.8,
            corner_radius=0.05,
        ).next_to(bottleneck, RIGHT, buff=0.5, aligned_edge=DOWN)
        decod_2 = RoundedRectangle(
            width=0.5,
            height=2.5,
            color=BLUE_C,
            fill_color=BLUE_C,
            fill_opacity=0.8,
            corner_radius=0.05,
        ).next_to(decod_3, RIGHT, buff=0.5, aligned_edge=DOWN)
        decod_1 = RoundedRectangle(
            width=0.5,
            height=3,
            color=BLUE_C,
            fill_color=BLUE_C,
            fill_opacity=0.8,
            corner_radius=0.05,
        ).next_to(decod_2, RIGHT, buff=0.5, aligned_edge=DOWN)

        skip_1 = Arrow(
            start=encod_1.get_corner(UR),
            end=decod_1.get_corner(UL),
            buff=0.1,
            color=WHITE,
        ).shift(0.25 * DOWN)
        skip_2 = Arrow(
            start=encod_2.get_corner(UR),
            end=decod_2.get_corner(UL),
            buff=0.1,
            color=WHITE,
        ).shift(0.25 * DOWN)
        skip_3 = Arrow(
            start=encod_3.get_corner(UR),
            end=decod_3.get_corner(UL),
            buff=0.1,
            color=WHITE,
        ).shift(0.25 * DOWN)
        arrows = VGroup(skip_1, skip_2, skip_3)
        unet = VGroup(
            encod_1,
            encod_2,
            encod_3,
            bottleneck,
            decod_1,
            decod_2,
            decod_3,
        )
        VGroup(unet, arrows).scale_to_fit_width(8).move_to(ORIGIN)

        self.play(LaggedStart(Write(title), Create(title_ul), lag_ratio=0.2))
        self.play(
            LaggedStart(
                FadeIn(unet),
                GrowArrow(skip_1),
                GrowArrow(skip_2),
                GrowArrow(skip_3),
                lag_ratio=0.2,
            ),
            run_time=2,
        )

        input_img = ImageMobject("./img/ffhq_3_noise_30.png").scale_to_fit_width(2)
        input_img.add(SurroundingRectangle(input_img, color=WHITE, buff=0.0))

        circle_t = Circle(
            radius=0.3,
            color=WHITE,
            fill_opacity=0,
        ).next_to(input_img, DOWN, buff=0.5)
        circle_t.add(
            MathTex(
                r"t",
                font_size=48,
                color=WHITE,
            ).move_to(circle_t.get_center())
        )

        inputs = (
            Group(input_img, circle_t)
            .arrange(DOWN, buff=0.5)
            .next_to(unet, LEFT, buff=0.75, aligned_edge=DOWN)
        )
        arrow_in_1 = Arrow(
            start=input_img.get_right(),
            end=(unet.get_left()[0], input_img.get_center()[1], 0),
            buff=0.1,
            color=WHITE,
        )
        arrow_in_2 = Arrow(
            start=circle_t.get_right(),
            end=(unet.get_left()[0], circle_t.get_center()[1], 0),
            buff=0.1,
            color=WHITE,
        )

        output = MathTex(
            r"s_{\theta}(x, t)", font_size=48, tex_to_color_map=tex_to_color_map
        )
        output.add(SurroundingRectangle(output, color=WHITE, buff=0.2))
        output.next_to(unet, RIGHT, buff=0.75)
        output_arrow = Arrow(
            start=unet.get_right(),
            end=output.get_left(),
            buff=0.1,
            color=WHITE,
        )

        self.play(FadeIn(input_img, arrow_in_1))
        self.play(LaggedStart(FadeIn(circle_t), GrowArrow(arrow_in_2), lag_ratio=0.2))
        self.play(
            LaggedStart(GrowArrow(output_arrow), FadeIn(output), lag_ratio=0.8),
            run_time=2,
        )

        self.play(FadeOut(title, title_ul))

        question = Tex(
            "How do we train this network?",
            font_size=42,
            color=WHITE,
        ).to_edge(UP, buff=1.0)
        question_ul = Underline(question, color=WHITE, buff=0.1)

        self.play(LaggedStart(Write(question), Create(question_ul), lag_ratio=0.2))

        self.play(
            FadeOut(
                unet,
                arrows,
                inputs,
                arrow_in_1,
                arrow_in_2,
                output,
                output_arrow,
                shift=0.5 * RIGHT,
            )
        )

        # The training objective
        self.next_section(skip_animations=False)

        objective = MathTex(
            r"s_{\theta}(x, t) \approx \nabla_x \log p_t(x)",
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        )
        obective_rect = SurroundingRectangle(objective, color=WHITE, buff=0.2)

        self.play(LaggedStart(Write(objective), Create(obective_rect), lag_ratio=0.8))

        loss = MathTex(
            r"\mathcal{L}(\theta) = \mathbb{E}_{x_0 \sim p(x), t} \left[ \| s_{\theta}(x, t) - \nabla_x \log p_t(x) \|^2 \right]",
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        )

        self.play(
            LaggedStart(
                VGroup(objective, obective_rect).animate.next_to(title, DOWN, buff=0.5),
                Write(loss),
                lag_ratio=0.3,
            ),
            run_time=3,
        )
        rect_score = SurroundingRectangle(loss[5], color=PURPLE_B, buff=0.1)

        score_label = Tex("Still unknown", font_size=36, color=RED).next_to(
            rect_score, DOWN, buff=0.2
        )

        self.play(Create(rect_score))
        self.play(FadeIn(score_label))

        self.play(ApplyWave(score_label, run_time=2.0))
        self.play(ApplyWave(score_label, run_time=2.0))

        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeOut(score_label, shift=DOWN),
                    FadeOut(rect_score, objective, obective_rect),
                ),
                loss.animate.next_to(title, DOWN, buff=0.5),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        conditional_loss = MathTex(
            r"\mathcal{L}(\theta) = \mathbb{E}_{x_0 \sim p(x), t} \left[ \| s_{\theta}(x, t) - \nabla_x \log p_t(x_t | x_0) \|^2 \right]",
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        )
        arrow = Arrow(
            start=loss.get_bottom(),
            end=conditional_loss.get_top(),
            buff=0.1,
            color=WHITE,
        )
        self.play(
            LaggedStart(
                GrowArrow(arrow),
                FadeIn(conditional_loss),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        # How it simplifies in the case of the DDPM SDE
        self.next_section(skip_animations=False)

        noisy_expression = MathTex(
            r"x_{t} = \sqrt{1 - \beta_{t}}x_{0} + \sqrt{\beta_{t}} \epsilon",
            font_size=42,
            color=WHITE,
        )
        ddpm_cond_dist = MathTex(
            r"p_{t}(x_{t} \mid x_{0}) = \mathcal{N}(\sqrt{1 - \beta_{t}} x_0, \beta_{t})",
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        )

        VGroup(noisy_expression, ddpm_cond_dist).arrange(RIGHT, buff=2.0).next_to(
            conditional_loss, UP, buff=1.5
        )
        arrow_2 = Arrow(
            start=noisy_expression.get_right(),
            end=ddpm_cond_dist.get_left(),
            buff=0.2,
            color=WHITE,
        )

        self.play(
            LaggedStart(
                FadeOut(loss, arrow, question, question_ul),
                Write(noisy_expression),
                lag_ratio=0.3,
            )
        )

        self.play(LaggedStart(GrowArrow(arrow_2), Write(ddpm_cond_dist), lag_ratio=0.3))

        score_ddpm = MathTex(
            r"\nabla_x \log p_t(x_t | x_0) =",
            r"\frac{\sqrt{1 - \beta_t}x_0 - x_t}{\beta_t}",
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        ).next_to(conditional_loss, UP, buff=0.75)
        score_ddpm_2 = MathTex(
            r"\nabla_x \log p_t(x_t | x_0) =",
            r"- \frac{\epsilon}{\beta_t}",
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        ).move_to(score_ddpm, aligned_edge=LEFT)

        self.play(
            LaggedStart(
                VGroup(noisy_expression, ddpm_cond_dist, arrow_2).animate.to_edge(
                    UP, buff=0.5
                ),
                Write(score_ddpm),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.play(
            Transform(score_ddpm[1:], score_ddpm_2[1:]),
            run_time=2,
        )

        final_loss = MathTex(
            r"\mathcal{L}(\theta) = \mathbb{E}_{x_0 \sim p(x), t} \left[ \| s_{\theta}(x, t) + \frac{\epsilon}{\beta_t} \|^2 \right]",
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        ).move_to(ORIGIN)

        self.play(
            LaggedStart(
                FadeOut(score_ddpm, arrow_2, noisy_expression, ddpm_cond_dist),
                ReplacementTransform(conditional_loss, final_loss),
                lag_ratio=0.3,
            )
        )

        final_loss_rect = SurroundingRectangle(final_loss, color=WHITE, buff=0.2)
        final_loss_label = Tex(
            "Denoising Score Matching loss", font_size=42, color=WHITE
        ).next_to(final_loss_rect, UP, buff=0.5)

        self.play(
            LaggedStart(
                Create(final_loss_rect),
                Write(final_loss_label),
                lag_ratio=0.3,
            )
        )
        ddpm_loss = MathTex(
            r"\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{x_0 \sim p(x), t} \left[ \| \epsilon_{\theta}(x, t) - \epsilon \|^2 \right]",
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        )
        ddpm_loss_rect = SurroundingRectangle(ddpm_loss, color=WHITE, buff=0.2)
        ddpm_loss_label = Tex("DDPM Loss", font_size=42, color=WHITE).next_to(
            ddpm_loss_rect, DOWN, buff=0.5
        )

        self.play(
            LaggedStart(
                VGroup(final_loss, final_loss_rect, final_loss_label).animate.to_edge(
                    UP, buff=0.5
                ),
                AnimationGroup(
                    Write(ddpm_loss),
                    Create(ddpm_loss_rect),
                    Write(ddpm_loss_label),
                ),
                lag_ratio=0.3,
            ),
            run_time=2,
        )

        self.play(
            VGroup(ddpm_loss, ddpm_loss_rect, ddpm_loss_label).animate.shift(1.5 * DOWN)
        )

        equivalence = MathTex(
            r"s_{\theta}(x, t) \approx - \epsilon_{\theta}(x, t)",
            font_size=42,
        )

        self.play(Write(equivalence))

        self.play(
            FadeOut(
                *self.mobjects,
            )
        )

        self.wait(1)


if __name__ == "__main__":
    scene = Scene1_8()
    scene.render()
