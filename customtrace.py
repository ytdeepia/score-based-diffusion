from __future__ import annotations
from manim import *

from typing import Callable
from manim import rate_functions


# Courtesy of Uwezi on the Manim Discord server
class Trace(VGroup):
    def __init__(
        self,
        traced_point_func: Callable,
        stroke_width: float = 5,
        stroke_width_func: Callable = rate_functions.linear,
        stroke_opacity_func: Callable = rate_functions.linear,
        stroke_color: color = WHITE,
        dissipating_time: float | None = None,
        **kwargs,
    ):
        super().__init__(stroke_color=stroke_color, stroke_width=stroke_width, **kwargs)
        self.traced_point_func = traced_point_func
        self.stroke_opacity_func = stroke_opacity_func
        self.stroke_width_func = stroke_width_func
        self.dissipating_time = dissipating_time
        self.ages = []
        self.time = 1 if self.dissipating_time else None
        self.add_updater(self.update_path)

    def update_path(self, mob, dt):
        new_point = self.traced_point_func()
        if len(self) == 0:
            self.submobjects.append(Dot(new_point, radius=0))
            self.ages.append(-dt)
        else:
            self.submobjects.append(
                Line(
                    start=self.submobjects[-1].get_end(),
                    end=new_point,
                    stroke_width=self.stroke_width,
                    stroke_color=self.stroke_color,
                )
            )
            self.ages.append(-dt)
            keep = 0
            # oldest points are at the start of the list
            for i in range(len(self.submobjects)):
                self.ages[i] += dt
                self.submobjects[i].set_stroke(
                    opacity=self.stroke_opacity_func(
                        1 - self.ages[i] / self.dissipating_time
                    ),
                    width=self.stroke_width_func(
                        1 - self.ages[i] / self.dissipating_time
                    )
                    * self.stroke_width,
                )
                if self.ages[i] >= self.dissipating_time:
                    keep = i
            self.submobjects = self.submobjects[keep:]
            self.ages = self.ages[keep:]
