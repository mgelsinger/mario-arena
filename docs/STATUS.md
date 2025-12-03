# Status & Notes

## Naming and layout
- Current structure is valid: repository folder `mario-arena`, Python package `mario_arena`.
- Imports rely on the `mario_arena/` package directory; the repo folder name is only for organization.
- Optional future refactor (not required): adopt a `src` layout to further separate package code:
```
mario-arena/
  src/
    mario_arena/
  tests/
  docs/
  setup.py
```

## Gymnasium migration
- Gym is unmaintained while Gymnasium is the maintained fork; Stable-Baselines3 already pulls Gymnasium.
- The current setup uses Gym 0.26.2 and compatibility shims, which cause the warnings you saw.
- Benefits of moving to Gymnasium: cleaner API, Python 3.11+ support, and removal of most compatibility code.
- Open question: does `gym-super-mario-bros` run under Gymnasium?
  - Options to test: run through `shimmy`, locate a Gymnasium-native Mario environment, or keep Gym just for the env while using Gymnasium elsewhere.
- Next step: test `gym-super-mario-bros` with Gymnasium (with `shimmy` if needed), then update dependencies and remove the compatibility wrappers based on the result.
