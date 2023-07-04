# sahil2801/replit-code-instruct-glaive Cog model

This is an implementation of the [sahil2801/replit-code-instruct-glaive](https://huggingface.co/sahil2801/replit-code-instruct-glaive) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="// javascript function that returns the meaning of life"
