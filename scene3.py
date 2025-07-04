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


class Scene1_3(MovingCameraScene):
    def construct(self):
        # Markov chain in DDPM
        self.next_section(skip_animations=False)

        xt_circle = Circle(radius=0.5, color=WHITE).move_to(RIGHT * 3.5)
        xt1_circle = Circle(radius=0.5, color=WHITE).move_to(LEFT * 3.5)

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

        markov_formula = MathTex(
            r"x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t",
            tex_to_color_map=tex_to_color_map,
            color=WHITE,
            font_size=32,
        )

        ddpm_diagram = VGroup(
            xt_circle, xt1_circle, xt_label, xt1_label, q_arrow, markov_formula
        )

        ffhq_30_img = (
            ImageMobject("./img/ffhq_3_noise_30.png")
            .scale_to_fit_width(2)
            .next_to(xt1_circle, UP, buff=1.5)
        )
        ffhq_30_img.add(SurroundingRectangle(ffhq_30_img, color=WHITE, buff=0.0))
        ffhq_30_line = DashedLine(
            start=xt1_circle.get_top(),
            end=ffhq_30_img.get_bottom(),
            color=WHITE,
        )
        ffhq_61_img = (
            ImageMobject("./img/ffhq_3_noise_61.png")
            .scale_to_fit_width(2)
            .next_to(xt_circle, UP, buff=1.5)
        )
        ffhq_61_img.add(SurroundingRectangle(ffhq_61_img, color=WHITE, buff=0.0))
        ffhq_61_line = DashedLine(
            start=xt_circle.get_top(),
            end=ffhq_61_img.get_bottom(),
            color=WHITE,
        )

        self.play(
            LaggedStart(
                FadeIn(xt1_circle, xt1_label),
                Create(ffhq_30_line),
                FadeIn(ffhq_30_img),
                lag_ratio=0.7,
                run_time=2,
            )
        )
        self.play(
            Create(q_arrow),
            run_time=1.5,
        )
        self.play(
            LaggedStart(
                FadeIn(xt_circle, xt_label),
                Create(ffhq_61_line),
                FadeIn(ffhq_61_img),
                lag_ratio=0.7,
                run_time=2,
            )
        )

        self.play(
            Write(markov_formula),
            run_time=1.5,
        )

        xtt_circle = Circle(radius=0.5, color=WHITE).next_to(xt_circle, RIGHT, buff=7.0)
        xtt_label = MathTex(
            r"x_{T}",
            color=WHITE,
            font_size=32,
        ).move_to(xtt_circle)
        q_arrow2 = DashedVMobject(
            CurvedArrow(
                start_point=xt_circle.get_top() + UP * 0.05,
                end_point=xtt_circle.get_top() + UP * 0.05,
                color=WHITE,
                angle=-PI / 2,
            )
        )
        pure_noise = (
            ImageMobject("./img/pure_noise.png")
            .scale_to_fit_width(2)
            .next_to(xtt_circle, UP, buff=1.5)
        )
        pure_noise.add(SurroundingRectangle(pure_noise, color=WHITE, buff=0.0))
        ffhq_102_line = DashedLine(
            start=xtt_circle.get_top(),
            end=pure_noise.get_bottom(),
            color=WHITE,
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(7 * RIGHT),
                Create(q_arrow2),
                lag_ratio=0.5,
                run_time=2,
            )
        )
        limit = MathTex(
            r"x_t \xrightarrow{t \to T} \mathcal{N}(0, I)",
            tex_to_color_map=tex_to_color_map,
            color=WHITE,
            font_size=32,
        ).move_to((q_arrow2.get_center()[0], markov_formula.get_center()[1], 0))
        self.play(
            LaggedStart(
                FadeIn(xtt_circle, xtt_label),
                Create(ffhq_102_line),
                FadeIn(pure_noise),
                lag_ratio=0.7,
                run_time=1.5,
            )
        )
        self.play(
            Write(limit),
            run_time=1.0,
        )

        self.play(
            LaggedStart(
                self.camera.frame.animate.shift(LEFT * 7),
                FadeOut(
                    q_arrow2, limit, xtt_label, xtt_circle, ffhq_102_line, pure_noise
                ),
                lag_ratio=0.5,
                run_time=2.0,
            )
        )

        q_arrow.add_updater(
            lambda m: m.put_start_and_end_on(xt1_circle.get_top(), xt_circle.get_top())
        )

        self.play(
            Group(xt1_circle, xt1_label, ffhq_30_img, ffhq_30_line).animate.shift(
                LEFT * 2.5
            ),
            Group(xt_circle, xt_label, ffhq_61_img, ffhq_61_line).animate.shift(
                RIGHT * 2.5
            ),
            run_time=1.75,
        )

        self.play(
            Group(xt1_circle, xt1_label, ffhq_30_img, ffhq_30_line).animate.shift(
                3 * RIGHT
            ),
            Group(xt_circle, xt_label, ffhq_61_img, ffhq_61_line).animate.shift(
                3 * LEFT
            ),
            run_time=1.75,
        )

        ito_sde = MathTex(
            r"\mathrm{d}x",
            r"=",
            r"f(x, t)",
            r"\mathrm{d}t",
            r"+",
            r"g(t)",
            r"\mathrm{d}W",
            tex_to_color_map=tex_to_color_map,
            color=WHITE,
            font_size=48,
            z_index=4,
        ).move_to(ORIGIN)
        self.play(
            LaggedStart(
                FadeOut(markov_formula),
                Group(xt1_circle, xt1_label, ffhq_30_img, ffhq_30_line).animate.shift(
                    2 * RIGHT
                ),
                Group(xt_circle, xt_label, ffhq_61_img, ffhq_61_line).animate.shift(
                    2 * LEFT
                ),
            ),
            run_time=2,
        )

        self.play(
            LaggedStart(
                FadeOut(ffhq_30_img, ffhq_30_line, ffhq_61_img, ffhq_61_line, q_arrow),
                ReplacementTransform(
                    VGroup(xt1_circle, xt1_label, xt_circle, xt_label), ito_sde
                ),
            )
        )

        # Ito SDE
        self.next_section(skip_animations=False)

        title = Tex(
            "Stochastic",
            " ",
            "Differential",
            " ",
            "Equation",
            color=WHITE,
            font_size=48,
        ).shift(2 * UP)

        self.play(
            Write(title),
            run_time=1.5,
        )

        diff_rect_title = SurroundingRectangle(title[2], color=WHITE, buff=0.1)
        diff_rect_sde_1 = SurroundingRectangle(ito_sde[0], color=WHITE, buff=0.1)
        diff_rect_sde_2 = SurroundingRectangle(ito_sde[3], color=WHITE, buff=0.1)

        self.play(Create(diff_rect_title))
        self.wait(1)
        self.play(ShowPassingFlash(diff_rect_sde_1, time_width=0.5, run_time=1))
        self.wait(1)
        self.play(ShowPassingFlash(diff_rect_sde_2, time_width=0.5, run_time=1))

        self.play(Uncreate(diff_rect_title))

        stoch_rect_title = SurroundingRectangle(title[0], color=WHITE, buff=0.1)
        stoch_rect_sde_1 = SurroundingRectangle(ito_sde[6], color=WHITE, buff=0.1)

        self.play(Create(stoch_rect_title))
        self.wait(2)
        self.play(ShowPassingFlash(stoch_rect_sde_1, time_width=0.5, run_time=1))
        self.wait(1)
        self.play(ShowPassingFlash(stoch_rect_sde_1, time_width=0.5, run_time=1))

        # 1D drifts
        self.next_section(skip_animations=False)

        self.play(
            LaggedStart(
                ito_sde.animate.to_edge(UP, buff=0.5),
                FadeOut(stoch_rect_title, title),
            )
        )

        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-2, 2, 1],
            x_length=10,
            y_length=4,
            axis_config={"color": WHITE},
        ).next_to(ito_sde, DOWN, buff=0.5)

        start_point = axes.c2p(-8, np.sin(-8))
        particle = Dot(color=BLUE_D, radius=0.075).move_to(start_point)
        v_arrow = Arrow(
            start=start_point,
            end=start_point + normalize(np.array([1, np.cos(start_point[0]), 0])),
            color=BLUE_D,
            buff=0,
            max_stroke_width_to_length_ratio=10,
            max_tip_length_to_length_ratio=0.25,
        )

        particle.add_updater(
            lambda m, dt: m.move_to(
                axes.c2p(
                    m.get_center()[0] + dt,
                    np.sin(m.get_center()[0]),
                )
            )
        )

        v_arrow.add_updater(
            lambda m, dt: m.put_start_and_end_on(
                particle.get_center(),
                particle.get_center()
                + normalize(
                    np.array([1, np.cos(axes.p2c(particle.get_center())[0]), 0])
                ),
            )
        )

        trajectory = axes.plot(
            lambda x: np.sin(x),
            x_range=[-5, 5],
            color=BLUE_C,
            stroke_width=4,
        )

        drift_label = Tex(r"drift", color=WHITE, font_size=46, z_index=3).next_to(
            ito_sde[2], DOWN, buff=0.25
        )
        drift_matrix = Matrix(
            [[1], [r"\cos(x)"]],
            element_to_mobject_config={"color": WHITE, "font_size": 46},
            element_alignment_corner=ORIGIN,
            z_index=3,
        ).next_to(ito_sde[2], DOWN, buff=0.25)

        self.add(particle, v_arrow)

        ito_sde_background_rect = SurroundingRectangle(
            ito_sde,
            color=WHITE,
            fill_color=BLACK,
            fill_opacity=0.95,
            buff=0.1,
            z_index=-1,
        )

        func = lambda pos: RIGHT + np.cos(pos[0]) * UP
        stream_lines = StreamLines(
            func,
            stroke_width=3,
            max_anchors_per_line=5,
            virtual_time=1,
            color=BLUE,
            z_index=-2,
        )
        stream_lines.start_animation(warm_up=False, flow_speed=0.75, time_width=0.5)

        self.play(FadeIn(ito_sde_background_rect, stream_lines))

        drift_label_background_rect = SurroundingRectangle(
            drift_label,
            color=WHITE,
            fill_color=BLACK,
            fill_opacity=0.95,
            buff=0.1,
            z_index=-1,
        )
        self.play(
            LaggedStart(
                FadeIn(drift_label_background_rect),
                Write(drift_label),
                lag_ratio=0.1,
            )
        )
        drift_matrix_background_rect = SurroundingRectangle(
            drift_matrix,
            color=WHITE,
            fill_color=BLACK,
            fill_opacity=0.95,
            buff=0.1,
            z_index=-1,
        )

        self.wait(3)
        self.play(
            ReplacementTransform(drift_label, drift_matrix),
            ReplacementTransform(
                drift_label_background_rect, drift_matrix_background_rect
            ),
        )

        self.play(
            FadeOut(
                drift_matrix, drift_matrix_background_rect, ito_sde_background_rect
            ),
            FadeOut(stream_lines),
        )

        particle.remove_updater(particle.updaters[0])
        self.remove(v_arrow)

        start_point = axes.c2p(0, 0)
        particle2 = Dot(color=RED_D, radius=0.075).move_to(start_point)

        self.play(Transform(particle, particle2))
        self.wait(2)
        self.play(Circumscribe(ito_sde[6], color=WHITE))
        self.wait(1)
        self.play(Circumscribe(ito_sde[5], color=WHITE))

        # Diffusion term
        self.next_section(skip_animations=False)

        diff_coeff = 0.1

        self.wait(1)

        particle.add_updater(
            lambda m, dt: m.move_to(
                axes.c2p(
                    axes.p2c(m.get_center())[0]
                    + np.random.normal(0, diff_coeff * dt**0.5),
                    axes.p2c(m.get_center())[1]
                    + np.random.normal(0, diff_coeff * dt**0.5),
                )
            )
        )
        diffusion_label = Tex(
            r"diffusion coefficient",
            color=RED_D,
            font_size=46,
        ).next_to(ito_sde[5], DOWN, buff=0.25)

        self.play(
            Write(diffusion_label),
        )

        diffusion = MathTex(r"0.1", color=RED_D, font_size=46).next_to(
            ito_sde[5], DOWN, buff=0.25
        )
        self.play(ReplacementTransform(diffusion_label, diffusion))
        self.wait(2)

        diff_coeff = 0.4

        diffusion2 = MathTex(r"0.4", color=RED_D, font_size=46).next_to(
            ito_sde[5], DOWN, buff=0.25
        )

        self.play(ReplacementTransform(diffusion, diffusion2))
        particle.remove_updater(particle.updaters[0])

        particle.add_updater(
            lambda m, dt: m.move_to(
                axes.c2p(
                    axes.p2c(m.get_center())[0]
                    + np.random.normal(0, diff_coeff * dt**0.5),
                    axes.p2c(m.get_center())[1]
                    + np.random.normal(0, diff_coeff * dt**0.5),
                )
            )
        )

        particle.remove_updater(particle.updaters[0])
        self.play(particle.animate.move_to(axes.c2p(0, 0)), FadeOut(diffusion2))

        # Both at the same time
        self.next_section(skip_animations=False)
        diff_coeff = 0.3
        particle.add_updater(
            lambda m, dt: m.move_to(
                axes.c2p(
                    axes.p2c(m.get_center())[0]
                    + dt
                    + np.random.normal(0, diff_coeff * dt**0.5),
                    np.sin(axes.p2c(m.get_center())[0])
                    + np.random.normal(0, diff_coeff * dt**0.5),
                )
            )
        )

        self.play(FadeOut(particle))

        # Multiple particles
        particles = VGroup()
        x_positions = np.linspace(-16, 8, 20)
        y_positions = np.linspace(-6, 1.25, 20)
        for x_pos in x_positions:
            for y_pos in y_positions:
                new_particle = Dot(color=RED_D, radius=0.05, z_index=2).move_to(
                    axes.c2p(x_pos, y_pos)
                )
                new_particle.add_updater(
                    lambda m, dt, y=y_pos: m.move_to(
                        axes.c2p(
                            axes.p2c(m.get_center())[0]
                            + dt
                            + np.random.normal(0, diff_coeff * dt**0.5),
                            y
                            + np.sin(axes.p2c(m.get_center())[0])
                            + np.random.normal(0, diff_coeff * dt**0.5),
                        )
                    )
                )
                particles.add(new_particle)

        self.play(FadeIn(particles), run_time=1)
        self.wait(2)
        func = lambda pos: RIGHT + np.cos(pos[0]) * UP
        stream_lines = StreamLines(
            func,
            stroke_width=3,
            max_anchors_per_line=5,
            virtual_time=1,
            color=BLUE,
            z_index=-2,
        )
        stream_lines.start_animation(warm_up=False, flow_speed=0.75, time_width=0.5)

        self.play(FadeIn(stream_lines))

        self.play(FadeOut(*self.mobjects))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene1_3()
    scene.render()
