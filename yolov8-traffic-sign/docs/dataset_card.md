# Dataset Card

## Task

Traffic sign detection.

## Expected label space

- go
- goForward
- goLeft
- stop
- stopLeft
- warning
- warningLeft

## Data format

Expected YOLO detection format:

- `images/train`, `images/val`, `images/test`
- `labels/train`, `labels/val`, `labels/test`
- dataset YAML describing paths and class names

## Notes

- Keep large raw datasets out of Git.
- Record any relabeling, redistribution, or day/night splits in this file.
- Add 3 to 5 example images to `assets/sample_predictions/` for presentation.
