# replit/replit-code-v1-3b Cog model

This is an implementation of the [replit/replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="// javascript function that returns the meaning of life"
