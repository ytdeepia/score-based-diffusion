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


class Scene1_9(Scene):
    def construct(self):

        # Discretizing the reverse SDE
        self.next_section(skip_animations=False)

        ddpm_reverse_sde = MathTex(
            r"\mathrm{d}x",
            r"= \left[-\frac{1}{2} \beta(t) x - \beta(t) \nabla_x \log p_t(x) \right] \mathrm{d}t + \sqrt{\beta(t)} \mathrm{d}\bar{W}",
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        )

        title = Tex("Euler-Maruyama Discretization", font_size=36).to_edge(UP, buff=0.5)
        title_ul = Underline(title, color=WHITE)

        ddpm_reverse_sde_network = MathTex(
            r"\mathrm{d}x",
            r"= \left[-\frac{1}{2} \beta(t) x - \beta(t) s_{\theta}(x, t) \right] \mathrm{d}t + \sqrt{\beta(t)} \mathrm{d}\bar{W}",
            font_size=42,
            tex_to_color_map=tex_to_color_map,
        ).move_to(ddpm_reverse_sde, aligned_edge=LEFT)

        sde_discretized_1 = MathTex(
            r"x_{t+1} - x_t",
            r"= \left[-\frac{1}{2} \beta(t) x_t - \beta(t) s_{\theta}(x_t, t) \right] \mathrm{d}t + \sqrt{\beta(t)} \mathrm{d}\bar{W}",
            font_size=42,
            tex_to_color_map=tex_to_color_map_2,
        )
        sde_discretized_2 = MathTex(
            r"x_{t+1}",
            r"= x_t + \left[-\frac{1}{2} \beta(t) x_t - \beta(t) s_{\theta}(x_t, t) \right] \Delta t + \sqrt{\beta(t)}",
            r"\mathrm{d}\bar{W}",
            font_size=42,
            tex_to_color_map=tex_to_color_map_2,
        ).move_to(sde_discretized_1, aligned_edge=LEFT)
        sde_discretized_3 = MathTex(
            r"x_{t+1}",
            r"= x_t + \left[-\frac{1}{2} \beta(t) x_t - \beta(t) s_{\theta}(x_t, t) \right] \Delta t + \sqrt{\beta(t)}",
            r"\sqrt{\Delta t} \epsilon",
            font_size=42,
            tex_to_color_map=tex_to_color_map_2,
        ).move_to(sde_discretized_1, aligned_edge=LEFT)

        self.play(Write(ddpm_reverse_sde))

        self.wait(3)
        self.play(Transform(ddpm_reverse_sde[1:], ddpm_reverse_sde_network[1:]))

        self.play(
            LaggedStart(Write(title), Create(title_ul), lag_ratio=0.2, run_time=1.5)
        )

        self.play(ddpm_reverse_sde.animate.next_to(title, DOWN, buff=0.75))
        self.play(Write(sde_discretized_1))

        self.play(Transform(sde_discretized_1, sde_discretized_2))
        self.play(Transform(sde_discretized_1[-1:], sde_discretized_3[-3:]))
        rect = SurroundingRectangle(sde_discretized_1, color=WHITE, buff=0.3)
        self.play(LaggedStart(FadeOut(ddpm_reverse_sde), Create(rect), lag_ratio=0.2))

        self.play(FadeOut(*self.mobjects))

        self.wait(1)


if __name__ == "__main__":
    scene = Scene1_9()
    scene.render()
