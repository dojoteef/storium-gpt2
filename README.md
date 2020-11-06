<div align="center">
  <img src="https://storium.cs.umass.edu/static/figment.svg">
</div>

# Storium GPT-2 Models

This is the official repository for the GPT-2 models described in the EMNLP
2020 paper *[STORIUM: A Dataset and Evaluation Platform for Machine-in-the-Loop
Story Generation]*. It has all the code necessary to reproduce the models and
analysis from the paper.

## Overview

<p>
<img src="https://storium.cs.umass.edu/static/overview.svg">
</p>

A high-level outline of our dataset and platform. In this example from a real
[STORIUM](https://storium.com/game/night-witches--2/act-1/scene-2) game, the
character ADIRA MAKAROVA uses the strength card DEADLY AIM to DISRUPT THE
GERMANS, a challenge card. Our model conditions on the natural language
annotations in the scene intro, challenge card, strength card, and character,
along with the text of the previous scene entry (not shown) to generate a
suggested story continuation. Players may then edit the model output, by adding
or deleting text, before publishing the entry. We collect these edits, using
the matched text as the basis of our USER metric. New models can be added to
the platform by simply implementing four methods: startup, shutdown,
preprocess, and generate.


## Deployment

This repository contains the code that makes our GPT-2 story generation models
deployable on our [evaluation platform](https://storium.cs.umass.edu), so it
serves as a great template for how to structure your code.  Please see the file
[figmentate.py](figmentate.py) for the simple API required for making your
model deployable on our platform. You will also need to provide a json file
with any properties needed to pass to your startup method. See for example the
properties below:


```json
{
  "scene_entry":
  {
    "properties": {
      "checkpoint_path": "/var/lib/figmentator/checkpoint",
      "sample": {
	"top_p": 0.9,
	"temperature": 0.9,
	"repetition_penalty": 1.2
      }
    },
    "requires": ["torch==1.3.0", "transformers==2.2.0", "kiwisolver==1.1.0"],
    "cls": "model=figmentate:GPT2Figmentator"
  }
}
```

The key *scene_entry* defines the type of model being created. Currently, we
only support models that generate the text of a scene entry, though we might
support other types of prediction models in the future, like suggesting cards
or narrator actions.

The *properties* object will be passed to your startup method. It allows for
defining any parameters needed for sampling from your model.

The *requires* list, is simply a list of python packages that need to be
installed for your model to run. These will be automatically installed when
your model is deployed. If you notice, we specify the deep learning package
[torch](https://pytorch.org) as a requirement. That's because our code is
agnostic to the underlying deep learning framework being used by your model.
That means it should support models using other frameworks like
[tensorflow](https://tensorflow.org) or [jax](https://github.com/google/jax).

Finally, the *cls* string is the class that wraps your model. It is specified
using Python's [entry
points](https://packaging.python.org/specifications/entry-points/#data-model)
syntax.

## Cite

```bibtex
@inproceedings{akoury2020storium,
  Author = {Nader Akoury, Shufan Wang, Josh Whiting, Stephen Hood, Nanyun Peng and Mohit Iyyer},
  Booktitle = {Empirical Methods for Natural Language Processing},
  Year = "2020",
  Title = {{STORIUM}: {A} {D}ataset and {E}valuation {P}latform for {S}tory {G}eneration}
}
```
