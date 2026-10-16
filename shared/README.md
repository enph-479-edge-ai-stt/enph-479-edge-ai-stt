# shared/

Cross-vertical specs and golden vectors go here.

- `specs/weight_image.md`: how a model's weights are laid out for the fabric (fixed-point formats, memory map, stream order), for the acoustic model and the character model. Provisional until `hardware/` builds to it.

- `golden/decode/`: the beam search's golden case, which training's and the runtime's copies of the decoder must both reproduce. See its README.

Still living in the training code, provisional: the feature spec (`training/src/training/features/fbank.py`) and the vocab (`training/src/training/vocab.py`; the runtime keeps a copy, checked against the decode golden case).
