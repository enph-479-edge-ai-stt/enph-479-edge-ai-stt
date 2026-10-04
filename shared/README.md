# shared/

Cross-vertical specs and golden vectors go here.

- `specs/am_weight_image.md`: how the acoustic model's weights are laid out for the fabric (fixed-point formats, memory map, stream order). Provisional until `hardware/` builds to it.

Still living in the training code, provisional: the feature spec (`training/src/training/features/fbank.py`) and the vocab (`training/src/training/vocab.py`). No golden vectors yet.
