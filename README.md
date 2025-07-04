# Score-based Diffusion Models

## What is this repo

This repository contains all the code used to generate the animations for the video ["Score-based Diffusion Models"](https://www.youtube.com/watch?v=lUljxdkolK8) on the youtube channel [Deepia](https://www.youtube.com/@Deepia-ls2fo).

All references to the script and the voiceover service have been removed, so you are left with only raw animations.

There is no guarantee that any of these scripts can actually run. The code was not meant to be re-used, so it might need some external data in some places that I may or may not have put in this repository.

You can reuse any piece of code to make the same visualizations, crediting the youtube channel [Deepia](https://www.youtube.com/@Deepia-ls2fo) would be nice but is not required.

## Environment

I switched to ``uv`` to manage my environments and recommend using it.
You can create a ``.venv`` from the requirements using these commands:
```bash
uv venv .manim_venv
source .manim_venv/bin/activate  
uv pip install -r requirements.txt
```

## Generate a scene

You should move to the animation subfolder then run the regular Manim commands to generate a video such as:

```bash
manim -pqh scene_1.py --disable_caching
```

The video will then be written in the ``./media/videos/scene_1/1080p60/`` subdirectory.

I recommand the ``--disable_caching`` flag when using voiceover.
